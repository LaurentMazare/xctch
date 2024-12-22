use xctch::{Context, Result};

fn main() -> Result<()> {
    xctch::et_pal_init();

    let program = xctch::Program::from_file("moshi-lm.pte")?;
    let mut method = program.method("forward")?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());
    for idx in 0..method.outputs_size() {
        println!("  out {idx}: {:?}", method.get_output(idx).tag())
    }
    for idx in 0..20 {
        let mut tensor = xctch::Tensor::from_data_with_dims(vec![0i64; 17], &[1, 17, 1])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().context("not a tensor")?;
        println!("out {idx} {:?}", logits.shape());
        let logits = logits.as_slice::<f32>().context("expected f32")?;
        println!("  {:?}", &logits[..10]);
    }
    println!();
    Ok(())
}
