use anyhow::Result;

fn main() -> Result<()> {
    xctch::et_pal_init();
    let program = xctch::Program::from_file("scripts/nanogpt/nanogpt.pte")?;
    let mut method = program.method("forward")?;
    let mut tensor = xctch::Tensor::from_data(vec![1i64]);
    println!("{}", tensor.nbytes());
    let evalue = tensor.as_evalue();
    method.set_input(&evalue, 0)?;
    method.set_input(&evalue, 1)?;
    unsafe { method.execute()? };
    let out = method.get_output(0);
    println!("{}", out.is_tensor());
    let out = out.as_tensor().unwrap();
    println!("{:?}", out.scalar_type());
    let out = out.as_slice::<f32>().unwrap().to_vec();
    println!("{out:?}");
    Ok(())
}
