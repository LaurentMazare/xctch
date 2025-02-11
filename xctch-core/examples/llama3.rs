use rand::{distributions::Distribution, SeedableRng};
use std::collections::HashMap;
use std::io::Write;
use xctch::{Context, Error as E, Result};

pub const TEMPERATURE: f32 = 0.6;
pub const SEED: u64 = 4242424242424242;

#[derive(clap::Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, value_name = "FILE", default_value = "llama3_2.pte")]
    pte: String,

    #[arg(long, value_name = "FILE", default_value = "scripts/llama3/vocab.json")]
    vocab: String,

    #[arg(short, long, default_value = "false")]
    verbose: bool,

    #[arg(short, long, default_value = "100")]
    n: usize,

    #[arg(long)]
    etdump: Option<String>,
}

struct Tokenizer {
    vocab: HashMap<usize, String>,
}

impl Tokenizer {
    fn token_str(&self, token_id: usize) -> String {
        self.vocab.get(&token_id).map_or("UNK", |v| v.as_str()).replace('Ġ', " ").replace("Ċ", "\n")
    }

    fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let json_data = std::fs::read_to_string(p)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&json_data).map_err(E::wrap)?;
        let vocab: HashMap<usize, String> = vocab.into_iter().map(|(k, v)| (v, k)).collect();
        Ok(Self { vocab })
    }
}

fn main() -> Result<()> {
    xctch::et_pal_init();

    let cli = <Cli as clap::Parser>::parse();

    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let tokenizer = Tokenizer::load(&cli.vocab).map_err(|e| e.with_path(&cli.vocab))?;

    let program = xctch::Program::from_file(&cli.pte)?;
    let mut method = program.method_d("forward")?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());
    for idx in 0..method.outputs_size() {
        println!("  out {idx}: {:?}", method.get_output(idx).tag())
    }
    let mut tokens = vec![22691i64];
    for &token in tokens.iter() {
        print!("{}", tokenizer.token_str(token as usize));
    }
    println!();
    for idx in 0..cli.n {
        let start_time = std::time::Instant::now();
        let tokens_len = tokens.len();
        let mut tensor = xctch::Tensor::from_data_with_dims(vec![tokens[tokens_len - 1]], &[1, 1])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        let mut tensor_pos = xctch::Tensor::from_data_with_dims(vec![idx as i64], &[1])?;
        let evalue_pos = tensor_pos.as_evalue();
        method.set_input(&evalue_pos, 1)?;
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().context("not a tensor")?;
        let logits = logits.as_slice::<half::bf16>().context("expected bf16")?;
        let token = if TEMPERATURE <= 0. {
            logits
                .iter()
                .enumerate()
                .max_by(|&(_, a), &(_, b)| a.total_cmp(b))
                .context("empty logits")?
                .0
        } else {
            let max_logit =
                logits.iter().max_by(|&a, &b| a.total_cmp(b)).context("empty logits")?;
            let mut sm: Vec<f32> =
                logits.iter().map(|v| ((*v - *max_logit).to_f32() / TEMPERATURE).exp()).collect();
            let sum_sm = sm.iter().sum::<f32>();
            sm.iter_mut().for_each(|v| *v /= sum_sm);
            let distr = rand::distributions::WeightedIndex::new(sm).map_err(E::wrap)?;
            distr.sample(&mut rng)
        };
        if !cli.verbose {
            print!("{}", tokenizer.token_str(token));
            std::io::stdout().flush()?;
        } else {
            let ms = start_time.elapsed().as_millis();
            println!("{idx:4}    token {token:5} {:12}     {ms}ms", tokenizer.token_str(token));
        }
        tokens.push(token as i64)
    }
    println!();
    if let Some(etdump_file) = cli.etdump.as_ref() {
        let dump_data = method.dump_data();
        std::fs::write(etdump_file, dump_data)?
    }
    Ok(())
}
