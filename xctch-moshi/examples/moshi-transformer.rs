use xctch::{Context, Result};

#[derive(clap::Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(long, value_name = "FILE", default_value = "moshi-lm.pte")]
    pte: String,

    #[arg(short, long, default_value = "false")]
    verbose: bool,

    #[arg(short, long, default_value = "100")]
    n: usize,

    #[arg(long)]
    etdump: Option<String>,
}

fn main() -> Result<()> {
    xctch::et_pal_init();

    let cli = <Cli as clap::Parser>::parse();

    let pte_file = &cli.pte;
    println!("loading model file {pte_file}");
    let program = xctch::Program::from_file(pte_file)?;
    let mut method = program.method_d("forward")?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());
    for idx in 0..method.outputs_size() {
        println!("  out {idx}: {:?}", method.get_output(idx).tag())
    }
    for idx in 0..cli.n {
        let start_time = std::time::Instant::now();
        let mut tensor = xctch::Tensor::from_data_with_dims(vec![0i64; 17], &[1, 17, 1])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().context("not a tensor")?;
        let shape = logits.shape();
        let logits = logits.as_slice::<f32>().context("expected f32")?;
        if cli.verbose {
            println!("  {:?}", &logits[..10]);
        }
        let ms = start_time.elapsed().as_millis();
        println!("{idx:5} {shape:?} {ms}ms")
    }
    println!();
    if let Some(etdump_file) = cli.etdump.as_ref() {
        let dump_data = method.dump_data();
        std::fs::write(etdump_file, dump_data)?
    }
    Ok(())
}

