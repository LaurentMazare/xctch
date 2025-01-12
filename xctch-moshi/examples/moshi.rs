use anyhow::{Context, Result};
use candle::{IndexOp, Tensor};
use candle_transformers::models::mimi;
use std::sync::{Arc, Mutex};

const BRIA_CODES: &[u8] = include_bytes!("../../assets/bria.safetensors");

#[derive(clap::Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, value_name = "FILE", default_value = "moshi-lm.pte")]
    pte: String,

    #[arg(long)]
    etdump: Option<String>,

    /// The maximum number of steps to run for
    #[arg(short, long)]
    n: Option<usize>,

    #[arg(long)]
    tokenizer: Option<String>,
}

fn load_vocab<P: AsRef<std::path::Path>>(
    path: Option<P>,
) -> Result<std::collections::HashMap<i64, String>> {
    let file = match path {
        None => {
            let api = hf_hub::api::sync::Api::new()?;
            let api = api.model("lmz/moshi-swift".to_string());
            tracing::info!("retrieving tokenizer");
            let path = api.get("tokenizer_spm_48k_multi6_2.json")?;
            std::fs::File::open(path)?
        }
        Some(path) => std::fs::File::open(path)?,
    };
    let file = std::io::BufReader::new(file);
    let data: std::collections::HashMap<i64, String> = serde_json::from_reader(file)?;
    Ok(data)
}

fn _make_mimi() -> Result<mimi::encodec::Encodec> {
    tracing::info!("make mimi");
    let api = hf_hub::api::sync::Api::new()?;
    let model = api.model("kyutai/mimi".to_string());
    tracing::info!("retrieving weights");
    let model = model.get("model.safetensors")?;
    tracing::info!("retrieving weights done");
    let model = model.to_string_lossy();
    let mut model = mimi::load(&model, Some(8), &candle::Device::Cpu)?;
    tracing::info!("warming up the model");
    let fake_pcm = candle::Tensor::zeros((1, 1, 1920), candle::DType::F32, &candle::Device::Cpu)?;
    let codes = model.encode_step(&fake_pcm.into())?;
    let pcm = model.decode_step(&codes)?;
    tracing::info!("warmed up model {:?}", pcm.shape());
    model.reset_state();
    Ok(model)
}

fn _model_encode(
    mut model: mimi::encodec::Encodec,
    rx: std::sync::mpsc::Receiver<Vec<f32>>,
) -> Result<()> {
    loop {
        let pcm = rx.recv()?;
        let start = std::time::Instant::now();
        let pcm_len = pcm.len();
        let pcm = Tensor::from_vec(pcm, (1, 1, pcm_len), &candle::Device::Cpu)?;
        let codes = model.encode_step(&pcm.into())?;
        tracing::info!("encode step {:?} in {:?}", codes.shape(), start.elapsed());
    }
}

fn _model_decode(mut model: mimi::encodec::Encodec, pcm_b: Arc<Mutex<Vec<f32>>>) -> Result<()> {
    tracing::info!("running model on {} threads", candle::utils::get_num_threads());
    let codes = candle::safetensors::load_buffer(BRIA_CODES, &candle::Device::Cpu)?;
    let codes = match codes.get("codes") {
        Some(tensor) => tensor.clone(),
        None => anyhow::bail!("cannot find codes"),
    };
    let len = codes.dim(candle::D::Minus1)?;
    for idx in 0..len {
        let start = std::time::Instant::now();
        let codes = codes.narrow(candle::D::Minus1, idx, 1)?;
        let pcm = model.decode_step(&codes.into())?;
        if let Some(pcm) = pcm.as_option() {
            let pcm = pcm.i(0)?.i(0)?;
            let mut pcm = pcm.to_vec1::<f32>()?;
            pcm_b.lock().unwrap().append(&mut pcm);
        }
        tracing::info!("decode step in {:?}", start.elapsed());
        let count = Arc::strong_count(&pcm_b);
        if count == 1 {
            tracing::info!("arc count is one, exiting thread");
            break;
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    xctch::et_pal_init();
    tracing_subscriber::fmt::init();

    let cli = <Cli as clap::Parser>::parse();

    let vocab = load_vocab(cli.tokenizer.as_ref())?;
    let pte_file = &cli.pte;
    tracing::info!("loading model file {pte_file}");
    let program = xctch::Program::from_file(pte_file)?;
    let mut method = program.method_d("forward")?;
    tracing::info!(
        "loaded method, inputs {}, outputs {}",
        method.inputs_size(),
        method.outputs_size()
    );
    for idx in 0..method.outputs_size() {
        tracing::info!("  out {idx}: {:?}", method.get_output(idx).tag())
    }
    let codes = candle::safetensors::load_buffer(BRIA_CODES, &candle::Device::Cpu)?;
    let codes = match codes.get("codes") {
        Some(tensor) => tensor.clone(),
        None => anyhow::bail!("cannot find codes"),
    };
    let len = codes.dim(candle::D::Minus1)?;
    let len = cli.n.map_or(len, |n| len.min(n));
    let mut last_token = 48000i64;
    {
        let mut tensor = vec![2048i64; 17];
        tensor[0] = last_token;
        let mut tensor = xctch::Tensor::from_data_with_dims(tensor, &[1, 17, 1])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        unsafe { method.execute()? };
    }
    for idx in 0..len {
        let start = std::time::Instant::now();
        let codes = codes.narrow(candle::D::Minus1, idx, 1)?;
        let codes = codes.flatten_all()?.to_vec1::<u32>()?;
        let mut tensor = vec![-1i64; 17];
        for (i, c) in codes.iter().enumerate() {
            tensor[i + 1] = *c as i64
        }
        if idx >= 25 {
            tensor[0] = last_token
        }
        let mut tensor = xctch::Tensor::from_data_with_dims(tensor, &[1, 17, 1])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().context("not a tensor")?;
        let logits = logits.as_slice::<f32>().context("expected f32")?;
        if let Some((token, _)) = logits.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b))
        {
            last_token = token as i64
        }
        let ms = start.elapsed().as_millis();
        let token = vocab.get(&last_token);
        tracing::info!(idx, ms, last_token, ?token)
    }
    if let Some(etdump_file) = cli.etdump.as_ref() {
        let dump_data = method.dump_data();
        std::fs::write(etdump_file, dump_data)?
    }
    Ok(())
}
