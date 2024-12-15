use anyhow::Result;

fn main() -> Result<()> {
    xctch::et_pal_init();
    let program = xctch::Program::from_file("/tmp/model.pte")?;
    let mut method = program.method("forward")?;
    let mut tensor = xctch::Tensor::from_data(vec![1.23f32]);
    println!("{}", tensor.nbytes());
    let evalue = tensor.as_evalue();
    method.set_input(&evalue, 0)?;
    method.set_input(&evalue, 1)?;
    unsafe { method.execute()? };
    let out = method.get_output(0);
    println!("{}", out.is_tensor());
    let out = out.as_tensor().unwrap();
    let mut dst = vec![-1f32];
    unsafe {
        std::ptr::copy_nonoverlapping(
            out.const_data_ptr() as *const f32,
            dst.as_mut_ptr(),
            dst.len(),
        );
    }
    println!("{dst:?}");
    Ok(())
}
