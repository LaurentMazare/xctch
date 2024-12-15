## xctch

Experimental bindings for executorch, currently compatible with version 0.4.

### Building executorch

```bash
mkdir cmake-out
cd cmake-out
cmake .. -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON
make -j8
```

### Running the nanogpt example

This is based on the [Into to LLMs in ExecuTorch](https://pytorch.org/executorch/0.4/llm/getting-started.html)
tutorial.

```bash
curl https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json -o scripts/nanogpt/vocab.json
curl https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py -o scripts/nanogpt/model.py
python scripts/nanogpt/export_nanogpt.py
```
