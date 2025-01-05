#pragma once

#include <executorch/runtime/core/portable_type/tensor_impl.h>
#include <executorch/runtime/core/portable_type/tensor.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/extension/data_loader/file_data_loader.h>
#include <executorch/extension/memory_allocator/malloc_memory_allocator.h>
#include <memory>

namespace torch {
namespace executor {

// https://github.com/dtolnay/cxx/issues/1049
using c_void = void;

std::unique_ptr<TensorImpl> tensor_impl(int32_t scalar_type, uint32_t ndims, int32_t* sizes, uint8_t* data);
std::unique_ptr<Tensor> tensor_new(TensorImpl&);
int32_t tensor_scalar_type(const Tensor&);

typedef Result<Method> ResultMethod;
typedef Result<MethodMeta> ResultMethodMeta;
typedef Result<Program> ResultProgram;
typedef Result<util::FileDataLoader> ResultFileDataLoader;
typedef Result<int64_t> ResultI64;
std::unique_ptr<Method> method_result_get(ResultMethod &v);
std::unique_ptr<MethodMeta> method_meta_result_get(ResultMethodMeta &v);
std::unique_ptr<Program> program_result_get(ResultProgram &v);
std::unique_ptr<util::FileDataLoader> file_data_loader_result_get(ResultFileDataLoader &v);
int64_t i64_result_get(ResultI64 &v);

std::unique_ptr<ResultProgram> program_load(util::FileDataLoader &);
std::unique_ptr<util::BufferDataLoader> buffer_data_loader_new(const void* data, size_t len);
std::unique_ptr<ResultProgram> program_load_b(util::BufferDataLoader &);
std::unique_ptr<ResultFileDataLoader> file_data_loader_from(std::string const&);
std::unique_ptr<MemoryManager> program_memory_manager_for_method(MethodMeta const& method_meta);
std::unique_ptr<ResultMethod> program_load_method(Program const& p, std::string const& name, MemoryManager &mgr);
std::unique_ptr<ResultMethodMeta> program_method_meta(Program const&, std::string const&);
std::unique_ptr<ResultI64> method_meta_memory_planned_buffer_size(MethodMeta const&, size_t);
std::unique_ptr<Tensor> evalue_to_tensor_move(EValue&);
const Tensor& evalue_to_tensor(const EValue&);
uint32_t evalue_tag(const EValue& e);
size_t evalue_str_len(const EValue&);
const char *evalue_str_ptr(const EValue&);
std::unique_ptr<EValue> evalue_from_tensor(Tensor&);
std::unique_ptr<EValue> evalue_from_double(double);
std::unique_ptr<EValue> evalue_from_int(int64_t);
uint32_t method_execute(Method&);
uint32_t method_set_input(Method&, const EValue&, size_t);
uint32_t method_set_output_data_ptr(Method&, void*, size_t, size_t);
}
}
