#include "cxx_api.hpp"

namespace torch {
namespace executor {

int32_t tensor_scalar_type(const Tensor& t) {
  return static_cast<int32_t>(t.scalar_type());
}

std::unique_ptr<TensorImpl> tensor_impl(int32_t scalar_type, uint32_t ndims, int32_t* sizes, uint8_t* data) {
  TensorImpl impl(
    static_cast<ScalarType>(scalar_type),
    ndims,
    sizes,
    data);
  return std::make_unique<TensorImpl>(std::move(impl));
}

std::unique_ptr<Tensor> tensor_new(TensorImpl& impl) {
  return std::make_unique<Tensor>(std::move(Tensor(&impl)));
}

template<typename T>
std::unique_ptr<T> result_get(Result<T> const &v) {
  return std::unique_ptr<T>(v.get());
}

std::unique_ptr<Method> method_result_get(ResultMethod &v) {
  return std::make_unique<Method>(std::move(v.get()));
}

std::unique_ptr<MethodMeta> method_meta_result_get(ResultMethodMeta &v) {
  return std::make_unique<MethodMeta>(std::move(v.get()));
}

std::unique_ptr<Program> program_result_get(ResultProgram &v) {
  return std::make_unique<Program>(std::move(v.get()));
}

std::unique_ptr<util::FileDataLoader> file_data_loader_result_get(ResultFileDataLoader &v) {
  return std::make_unique<util::FileDataLoader>(std::move(v.get()));
}

int64_t i64_result_get(ResultI64 &v) {
  return v.get();
}

std::unique_ptr<ResultFileDataLoader> file_data_loader_from(std::string const& path) {
  return std::make_unique<ResultFileDataLoader>(util::FileDataLoader::from(path.c_str()));
}

std::unique_ptr<ResultProgram> program_load(util::FileDataLoader& reader) {
  return std::make_unique<ResultProgram>(Program::load(&reader));
}

std::unique_ptr<MemoryManager> program_memory_manager_for_method(MethodMeta const& method_meta) {
  std::vector<std::unique_ptr<uint8_t[]>> *planned_buffers = new std::vector<std::unique_ptr<uint8_t[]>>(); // Owns the memory
  std::vector<torch::executor::Span<uint8_t>> *planned_spans = new std::vector<torch::executor::Span<uint8_t>>();
  size_t num_memory_planned_buffers = method_meta.num_memory_planned_buffers();

  for (size_t id = 0; id < num_memory_planned_buffers; ++id) {
    size_t buffer_size =
        static_cast<size_t>(method_meta.memory_planned_buffer_size(id).get());
    ET_LOG(
        Info, "ET: Setting up planned buffer %zu, size %zu.", id, buffer_size);

    planned_buffers->push_back(std::make_unique<uint8_t[]>(buffer_size));
    planned_spans->push_back({planned_buffers->back().get(), buffer_size});
  }


  util::MallocMemoryAllocator *method_allocator = new util::MallocMemoryAllocator();
  HierarchicalAllocator *planned_memory = new HierarchicalAllocator(
      {planned_spans->data(), planned_spans->size()});
  return std::make_unique<MemoryManager>(MemoryManager(method_allocator, planned_memory));
}

std::unique_ptr<ResultMethod> program_load_method(Program const& p, std::string const& name, MemoryManager &mgr) {
  return std::make_unique<ResultMethod>(p.load_method(name.c_str(), &mgr));
}

std::unique_ptr<ResultMethodMeta> program_method_meta(Program const& p, std::string const& name) {
  return std::make_unique<ResultMethodMeta>(p.method_meta(name.c_str()));
}

std::unique_ptr<ResultI64> method_meta_memory_planned_buffer_size(MethodMeta const& m, size_t idx) {
  return std::make_unique<ResultI64>(m.memory_planned_buffer_size(idx));
}

const Tensor& evalue_to_tensor(const EValue& e) {
  return e.toTensor();
}

uint32_t evalue_tag(const EValue& e) {
  return static_cast<uint32_t>(e.tag);
}

std::unique_ptr<Tensor> evalue_to_tensor_move(EValue& e) {
  return std::make_unique<Tensor>(e.toTensor());
}

std::unique_ptr<EValue> evalue_from_tensor(Tensor& e) {
  return std::make_unique<EValue>(std::move(EValue(std::move(e))));
}

std::unique_ptr<EValue> evalue_from_double(double e) {
  return std::make_unique<EValue>(EValue(e));
}

std::unique_ptr<EValue> evalue_from_int(int64_t e) {
  return std::make_unique<EValue>(EValue(e));
}

uint32_t method_execute(Method& m) {
  return static_cast<uint32_t>(m.execute());
}

uint32_t method_set_input(Method& m, const EValue& e, size_t idx) {
  return static_cast<uint32_t>(m.set_input(e, idx));
}

uint32_t method_set_output_data_ptr(Method& m, void* p, size_t sz, size_t idx) {
  return static_cast<uint32_t>(m.set_output_data_ptr(p, sz, idx));
}

}
}
