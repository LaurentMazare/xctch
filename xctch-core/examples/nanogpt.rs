use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;

fn main() -> Result<()> {
    xctch::et_pal_init();
    let file_path = "scripts/nanogpt/vocab.json";
    let json_data = std::fs::read_to_string(file_path)?;
    let vocab: HashMap<String, usize> = serde_json::from_str(&json_data)?;
    let vocab: HashMap<usize, String> = vocab.into_iter().map(|(k, v)| (v, k)).collect();

    let program = xctch::Program::from_file("scripts/nanogpt/nanogpt.pte")?;
    let mut method = program.method("forward")?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());
    let mut tokens = vec![18435i64];
    for &token in tokens.iter() {
        let token = token as usize;
        let token_str = vocab.get(&token).map_or("", |v| v.as_str()).replace('Ġ', " ");
        print!("{}", token_str);
    }
    for _ in 0..20 {
        let tokens_len = tokens.len();
        let mut tensor = xctch::Tensor::from_data_with_dims(tokens.clone(), &[1, tokens_len])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        // TODO: Add a version with KV cache.
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().unwrap();
        let logits = logits.as_slice::<f32>().unwrap();
        // TODO: softmax + sampling
        let token = logits.iter().enumerate().max_by(|&(_, a), &(_, b)| a.total_cmp(b)).unwrap().0;
        let token_str = vocab.get(&token).map_or("", |v| v.as_str()).replace('Ġ', " ");
        print!("{}", token_str);
        let _ = std::io::stdout().flush();
        tokens.push(token as i64)
    }
    Ok(())
}
