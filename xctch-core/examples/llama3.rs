use anyhow::Result;
use std::collections::HashMap;
use std::io::Write;

struct Tokenizer {
    vocab: HashMap<usize, String>,
}

impl Tokenizer {
    fn token_str(&self, token_id: usize) -> String {
        self.vocab.get(&token_id).map_or("UNK", |v| v.as_str()).replace('Ġ', " ").replace("Ċ", "\n")
    }

    fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let json_data = std::fs::read_to_string(p)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&json_data)?;
        let vocab: HashMap<usize, String> = vocab.into_iter().map(|(k, v)| (v, k)).collect();
        Ok(Self { vocab })
    }
}

fn main() -> Result<()> {
    xctch::et_pal_init();
    let tokenizer = Tokenizer::load("scripts/llama3/vocab.json")?;

    let program = xctch::Program::from_file("llama3_2.pte")?;
    let mut method = program.method("forward")?;
    println!("loaded method, inputs {}, outputs {}", method.inputs_size(), method.outputs_size());
    for idx in 0..method.outputs_size() {
        println!("  out {idx}: {:?}", method.get_output(idx).tag())
    }
    let mut tokens = vec![22691i64];
    for &token in tokens.iter() {
        print!("{}", tokenizer.token_str(token as usize));
    }
    for idx in 0..20 {
        let tokens_len = tokens.len();
        let mut tensor = xctch::Tensor::from_data_with_dims(tokens.clone(), &[1, tokens_len])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        let mut tensor_pos = xctch::Tensor::from_data_with_dims(vec![idx as i64], &[1])?;
        let evalue_pos = tensor_pos.as_evalue();
        method.set_input(&evalue_pos, 1)?;
        // TODO: Add a version with KV cache.
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().unwrap();
        let logits = logits.as_slice::<half::bf16>().unwrap();
        // TODO: softmax + sampling
        let token = logits.iter().enumerate().max_by(|&(_, a), &(_, b)| a.total_cmp(b)).unwrap().0;
        print!("{}", tokenizer.token_str(token as usize));
        std::io::stdout().flush()?;
        tokens.push(token as i64)
    }
    println!();
    Ok(())
}
