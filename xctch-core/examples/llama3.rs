use rand::{distributions::Distribution, SeedableRng};
use std::collections::HashMap;
use std::io::Write;
use xctch::{Error as E, Result};

pub const TEMPERATURE: f32 = 0.6;
pub const SEED: u64 = 4242424242424242;

struct Tokenizer {
    vocab: HashMap<usize, String>,
}

impl Tokenizer {
    fn token_str(&self, token_id: usize) -> String {
        self.vocab.get(&token_id).map_or("UNK", |v| v.as_str()).replace('Ġ', " ").replace("Ċ", "\n")
    }

    fn load<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let json_data = std::fs::read_to_string(p)?;
        let vocab: HashMap<String, usize> = serde_json::from_str(&json_data).map_err(E::wrap)?;
        let vocab: HashMap<usize, String> = vocab.into_iter().map(|(k, v)| (v, k)).collect();
        Ok(Self { vocab })
    }
}

fn main() -> Result<()> {
    xctch::et_pal_init();
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let vocab_path = "scripts/llama3/vocab.json";
    let tokenizer = Tokenizer::load(vocab_path).map_err(|e| e.with_path(vocab_path))?;

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
    for idx in 0..200 {
        let tokens_len = tokens.len();
        let mut tensor = xctch::Tensor::from_data_with_dims(vec![tokens[tokens_len - 1]], &[1, 1])?;
        let evalue = tensor.as_evalue();
        method.set_input(&evalue, 0)?;
        let mut tensor_pos = xctch::Tensor::from_data_with_dims(vec![idx as i64], &[1])?;
        let evalue_pos = tensor_pos.as_evalue();
        method.set_input(&evalue_pos, 1)?;
        unsafe { method.execute()? };
        let logits = method.get_output(0);
        let logits = logits.as_tensor().unwrap();
        let logits = logits.as_slice::<half::bf16>().unwrap();
        let token = if TEMPERATURE <= 0. {
            logits.iter().enumerate().max_by(|&(_, a), &(_, b)| a.total_cmp(b)).unwrap().0
        } else {
            let max_logit = logits.iter().max_by(|&a, &b| a.total_cmp(b)).unwrap();
            let mut sm: Vec<f32> =
                logits.iter().map(|v| ((*v - *max_logit).to_f32() / TEMPERATURE).exp()).collect();
            let sum_sm = sm.iter().sum::<f32>();
            sm.iter_mut().for_each(|v| *v /= sum_sm);
            let distr = rand::distributions::WeightedIndex::new(sm).map_err(E::wrap)?;
            distr.sample(&mut rng)
        };
        print!("{}", tokenizer.token_str(token));
        std::io::stdout().flush()?;
        tokens.push(token as i64)
    }
    println!();
    Ok(())
}
