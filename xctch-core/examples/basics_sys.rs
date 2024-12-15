use anyhow::Result;

fn main() -> Result<()> {
    use xctch_sys::safe;
    println!("hello world!");
    unsafe { xctch_sys::et_pal_init() };
    let mut fdl = safe::FileDataLoader::new("/tmp/model.pte")?;
    let program = safe::Program::load(&mut fdl)?;
    let method_meta = program.method_meta("forward")?;
    let mut mgr = method_meta.memory_manager();
    let mut method = program.method("forward", &mut mgr)?;
    let mut data = vec![1.23f32];
    let mut tensor_impl = safe::TensorImpl::from_data(&mut data);
    let mut tensor = safe::Tensor::new(&mut tensor_impl);
    println!("{}", tensor.nbytes());
    let evalue = safe::EValue::from_tensor(&mut tensor);
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
