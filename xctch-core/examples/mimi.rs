use anyhow::Result;

fn main() -> Result<()> {
    xctch::et_pal_init();

    let program = xctch::Program::from_file("mimi.pte")?;
    let pcm_len = 1920;
    let pcm = vec![0f32; 1920];

    let mut method = program.method("forward")?;
    let mut tensor = xctch::Tensor::from_data_with_dims(pcm.clone(), &[1, 1, pcm_len])?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());

    let evalue = tensor.as_evalue();
    method.set_input(&evalue, 0)?;
    for idx in 0..method.outputs_size() {
        println!("  out {idx}: {:?}", method.get_output(idx).tag())
    }
    println!();
    unsafe { method.execute()? };
    let out = method.get_output(0);
    println!("out: {:?}", out.tag());
    let out = out.as_tensor().unwrap();
    let out = out.as_slice::<i64>().unwrap();
    println!("{out:?}");
    Ok(())
}
