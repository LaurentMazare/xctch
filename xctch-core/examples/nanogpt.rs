use anyhow::Result;

fn main() -> Result<()> {
    xctch::et_pal_init();
    let program = xctch::Program::from_file("scripts/nanogpt/nanogpt.pte")?;
    let mut method = program.method("forward")?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());
    let mut tensor = xctch::Tensor::from_data_with_dims(vec![1i64], &[1, 1])?;
    println!("{}", tensor.nbytes());
    let evalue = tensor.as_evalue();
    method.set_input(&evalue, 0)?;
    unsafe { method.execute()? };
    let out = method.get_output(0);
    println!("{}", out.is_tensor());
    let out = out.as_tensor().unwrap();
    println!("{:?}", out.scalar_type());
    let out = out.as_slice::<f32>().unwrap();
    println!("{:?}", &out[..10]);
    let out2 = method.get_output(1);
    println!("{}", out2.is_tensor());
    Ok(())
}
