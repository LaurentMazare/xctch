use anyhow::Result;

fn main() -> Result<()> {
    xctch::et_pal_init();

    let pcm_len = 24000;
    let pcm = vec![0f32; pcm_len];

    let codes = {
        let program = xctch::Program::from_file("mimi_encoder.pte")?;
        let mut method = program.method("forward")?;
        let mut tensor = xctch::Tensor::from_data_with_dims(pcm.clone(), &[1, 1, pcm_len])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        unsafe { method.execute()? };
        let out = method.get_output(0);
        println!("out: {:?}", out.tag());
        let out = out.as_tensor().unwrap();
        println!("{}", out.dim());
        out.as_slice::<i64>().unwrap().to_vec()
    };
    println!("{codes:?}");
    let pcm = {
        let program = xctch::Program::from_file("mimi_decoder.pte")?;
        let mut method = program.method("forward")?;
        let mut tensor =
            xctch::Tensor::from_data_with_dims(codes.clone(), &[1, 8, codes.len() / 8])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        unsafe { method.execute()? };
        let out = method.get_output(0);
        println!("out: {:?}", out.tag());
        let out = out.as_tensor().unwrap();
        println!("{}", out.dim());
        out.as_slice::<f32>().unwrap().to_vec()
    };
    println!("{:?}", &pcm[..20]);
    Ok(())
}
