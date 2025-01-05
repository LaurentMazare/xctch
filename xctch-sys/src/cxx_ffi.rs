#![allow(clippy::missing_safety_doc)]

#[cxx::bridge(namespace = "torch::executor")]
pub mod ffi {
    unsafe extern "C++" {
        include!("xctch-sys/cpp/cxx_api.hpp");

        type c_void;

        pub type TensorImpl;
        pub type Tensor;
        pub type Program;
        pub type Method;
        pub type MethodMeta;
        pub type EValue;
        pub type MemoryManager;

        fn num_memory_planned_buffers(self: &MethodMeta) -> usize;
        fn method_meta_memory_planned_buffer_size(
            meta: &MethodMeta,
            idx: usize,
        ) -> UniquePtr<ResultI64>;
        fn nbytes(self: &Tensor) -> usize;
        fn dim(self: &Tensor) -> isize;
        fn size(self: &Tensor, dim: isize) -> isize;
        fn numel(self: &Tensor) -> isize;
        fn const_data_ptr(self: &Tensor) -> *const c_void;
        fn mutable_data_ptr(self: &Tensor) -> *mut c_void;
        fn tensor_scalar_type(t: &Tensor) -> i32;
        unsafe fn tensor_impl(
            scalar_type: i32,
            ndims: u32,
            dims: *mut i32,
            data: *mut u8,
        ) -> UniquePtr<TensorImpl>;
        fn tensor_new(t: Pin<&mut TensorImpl>) -> UniquePtr<Tensor>;
        fn num_methods(self: &Program) -> usize;
        fn inputs_size(self: &Method) -> usize;
        fn outputs_size(self: &Method) -> usize;
        fn num_inputs(self: &MethodMeta) -> usize;
        fn num_outputs(self: &MethodMeta) -> usize;

        fn isNone(self: &EValue) -> bool;
        fn isInt(self: &EValue) -> bool;
        fn isDouble(self: &EValue) -> bool;
        fn isScalar(self: &EValue) -> bool;
        fn isTensor(self: &EValue) -> bool;
        fn isString(self: &EValue) -> bool;
        fn isIntList(self: &EValue) -> bool;
        fn isDoubleList(self: &EValue) -> bool;
        fn isTensorList(self: &EValue) -> bool;
        fn toInt(self: &EValue) -> i64;
        fn toDouble(self: &EValue) -> f64;
        fn toBool(self: &EValue) -> bool;
        fn evalue_to_tensor(e: &EValue) -> &Tensor;
        fn evalue_to_tensor_move(e: Pin<&mut EValue>) -> UniquePtr<Tensor>;
        fn evalue_from_int(i: i64) -> UniquePtr<EValue>;
        fn evalue_from_double(f: f64) -> UniquePtr<EValue>;
        fn evalue_from_tensor(t: Pin<&mut Tensor>) -> UniquePtr<EValue>;
        fn evalue_tag(e: &EValue) -> u32;
        fn evalue_str_len(e: &EValue) -> usize;
        fn evalue_str_ptr(e: &EValue) -> *const c_char;

        fn program_load(loader: Pin<&mut FileDataLoader>) -> UniquePtr<ResultProgram>;
        fn program_load_b(loader: Pin<&mut BufferDataLoader>) -> UniquePtr<ResultProgram>;
        fn program_memory_manager_for_method(m: &MethodMeta) -> UniquePtr<MemoryManager>;
        fn program_load_method(
            p: &Program,
            name: &CxxString,
            mgr: Pin<&mut MemoryManager>,
        ) -> UniquePtr<ResultMethod>;
        fn program_method_meta(p: &Program, name: &CxxString) -> UniquePtr<ResultMethodMeta>;
        fn method_execute(m: Pin<&mut Method>) -> u32;
        fn method_set_input(m: Pin<&mut Method>, e: &EValue, idx: usize) -> u32;
        fn get_output(self: &Method, idx: usize) -> &EValue;
        unsafe fn method_set_output_data_ptr(
            m: Pin<&mut Method>,
            ptr: *mut c_void,
            size: usize,
            idx: usize,
        ) -> u32;

        #[namespace = "torch::executor::util"]
        pub type FileDataLoader;
        fn file_data_loader_from(path: &CxxString) -> UniquePtr<ResultFileDataLoader>;

        #[namespace = "torch::executor::util"]
        pub type BufferDataLoader;
        unsafe fn buffer_data_loader_new(
            data: *const c_void,
            l: usize,
        ) -> UniquePtr<BufferDataLoader>;

        pub type ResultMethodMeta;
        fn ok(self: &ResultMethodMeta) -> bool;
        fn method_meta_result_get(v: Pin<&mut ResultMethodMeta>) -> UniquePtr<MethodMeta>;

        pub type ResultMethod;
        fn ok(self: &ResultMethod) -> bool;
        fn method_result_get(v: Pin<&mut ResultMethod>) -> UniquePtr<Method>;

        pub type ResultProgram;
        fn ok(self: &ResultProgram) -> bool;
        fn program_result_get(v: Pin<&mut ResultProgram>) -> UniquePtr<Program>;

        pub type ResultFileDataLoader;
        fn ok(self: &ResultFileDataLoader) -> bool;
        fn file_data_loader_result_get(
            v: Pin<&mut ResultFileDataLoader>,
        ) -> UniquePtr<FileDataLoader>;

        pub type ResultI64;
        fn ok(self: &ResultI64) -> bool;
        fn i64_result_get(v: Pin<&mut ResultI64>) -> i64;
    }
}

extern "C" {
    pub fn et_pal_init();
}
