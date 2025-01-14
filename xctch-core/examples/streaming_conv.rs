use anyhow::Result;

fn main() -> Result<()> {
    xctch::et_pal_init();

    let pcm_len = 5;
    let pcm: Vec<f32> = (0..pcm_len).map(|v| (v as f32).cos()).collect();

    let program = xctch::Program::from_file("streaming_conv1d.pte")?;
    let mut forward = program.method("forward")?;
    let mut reset = program.method("reset")?;
    for idx in 0..6 {
        println!("Step {idx}");
        let mut tensor = xctch::Tensor::from_data_with_dims(pcm.clone(), &[1, 1, pcm_len])?;
        let evalue = tensor.as_evalue();
        forward.set_input(&evalue, 0)?;
        unsafe { forward.execute()? };
        let out = forward.get_output(0);
        println!("out: {:?}", out.tag());
        let out = out.as_tensor().unwrap();
        println!("{:?}", out.shape());
        let out = out.as_slice::<f32>().unwrap();
        println!("{out:?}");
        if idx == 3 {
            println!("reset");
            reset.set_input(&evalue, 0)?;
            unsafe { reset.execute()? };
        }
    }
    Ok(())
}
