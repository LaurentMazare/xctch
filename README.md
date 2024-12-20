## xctch

Experimental bindings for executorch, currently compatible with version 0.4.

### Building executorch
From the `executorch` directory:
```bash
rm -Rf cmake-out
cmake . \
  -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
  -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
  -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
  -DEXECUTORCH_ENABLE_LOGGING=ON \
  -DEXECUTORCH_BUILD_KERNELS_OPTIMIZED=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DEXECUTORCH_BUILD_XNNPACK=ON \
  -DCMAKE_INSTALL_PREFIX=cmake-out \
  -Bcmake-out
cmake --build cmake-out -j16 --target install --config Release
```

### Running the nanogpt example

This is based on the [Into to LLMs in ExecuTorch](https://pytorch.org/executorch/0.4/llm/getting-started.html)
tutorial.

```bash
curl https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json -o scripts/nanogpt/vocab.json
curl https://raw.githubusercontent.com/karpathy/nanoGPT/master/model.py -o scripts/nanogpt/model.py
python scripts/nanogpt/export_nanogpt.py
```

### Common Issues

```
E 00:00:00.000061 executorch:operator_registry.cpp:85] Re-registering aten::add.out, from NOT_SUPPORTED
E 00:00:00.000075 executorch:operator_registry.cpp:86] key: (null), is_fallback: true
F 00:00:00.000076 executorch:operator_registry.cpp:106] In function register_kernels(), assert failed (false): Kernel registration failed with error 18, see error log for details.
```
This is caused by the portable and optimized ops being linked together in the
final binary. These are supposed to be merged together
[here](https://github.com/pytorch/executorch/blob/e94c2ff279e07b62928900cba572eb2fea03feb4/configurations/CMakeLists.txt#L32)
and the binary shoud link with `optimized_native_cpu_ops_lib` rather
than the other `.*_ops_lib` files.
