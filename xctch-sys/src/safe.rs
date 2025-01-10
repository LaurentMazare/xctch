use crate::{ffi, Error, Result, Tag};
use std::marker::PhantomData;

pub struct BufferDataLoader<'a> {
    inner: cxx::UniquePtr<ffi::BufferDataLoader>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> BufferDataLoader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let inner =
            unsafe { ffi::buffer_data_loader_new(data.as_ptr() as *const ffi::c_void, data.len()) };
        Self { inner, _marker: PhantomData }
    }
}

pub struct FileDataLoader {
    inner: cxx::UniquePtr<ffi::FileDataLoader>,
}

trait Res: cxx::memory::UniquePtrTarget + Sized {
    type Inner: cxx::memory::UniquePtrTarget;
    fn ok(&self) -> bool;
    fn err(&self) -> crate::error::FfiError;
    fn get(v: &mut cxx::UniquePtr<Self>) -> cxx::UniquePtr<Self::Inner>;
}

impl Res for ffi::ResultMethodMeta {
    type Inner = ffi::MethodMeta;
    fn ok(&self) -> bool {
        self.ok()
    }
    fn err(&self) -> crate::error::FfiError {
        crate::error::FfiError::Internal
    }
    fn get(v: &mut cxx::UniquePtr<Self>) -> cxx::UniquePtr<Self::Inner> {
        ffi::method_meta_result_get(v.pin_mut())
    }
}

impl Res for ffi::ResultMethod {
    type Inner = ffi::Method;
    fn ok(&self) -> bool {
        self.ok()
    }
    fn err(&self) -> crate::error::FfiError {
        crate::error::FfiError::Internal
    }
    fn get(v: &mut cxx::UniquePtr<Self>) -> cxx::UniquePtr<Self::Inner> {
        ffi::method_result_get(v.pin_mut())
    }
}

impl Res for ffi::ResultProgram {
    type Inner = ffi::Program;
    fn ok(&self) -> bool {
        self.ok()
    }
    fn err(&self) -> crate::error::FfiError {
        crate::error::FfiError::Internal
    }
    fn get(v: &mut cxx::UniquePtr<Self>) -> cxx::UniquePtr<Self::Inner> {
        ffi::program_result_get(v.pin_mut())
    }
}

impl Res for ffi::ResultFileDataLoader {
    type Inner = ffi::FileDataLoader;
    fn ok(&self) -> bool {
        self.ok()
    }
    fn err(&self) -> crate::error::FfiError {
        crate::error::FfiError::Internal
    }
    fn get(v: &mut cxx::UniquePtr<Self>) -> cxx::UniquePtr<Self::Inner> {
        ffi::file_data_loader_result_get(v.pin_mut())
    }
}

fn to_result<R: Res>(r: &mut cxx::UniquePtr<R>) -> Result<cxx::UniquePtr<R::Inner>> {
    if r.is_null() {
        return Err(Error::NullFfiPtr.bt());
    }
    if !r.ok() {
        return Err(Error::FfiError(r.err()).bt());
    }
    Ok(Res::get(r))
}

impl FileDataLoader {
    pub fn new<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let p = p.as_ref().as_os_str();
        cxx::let_cxx_string!(p = p.as_encoded_bytes());
        let mut fdl = ffi::file_data_loader_from(&p);
        let fdl = to_result(&mut fdl)?;
        Ok(Self { inner: fdl })
    }
}

pub struct Program<'a> {
    inner: cxx::UniquePtr<ffi::Program>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Program<'a> {
    pub fn load_b(bdl: &'a mut BufferDataLoader) -> Result<Self> {
        let mut program = ffi::program_load_b(bdl.inner.as_mut().unwrap());
        let program = to_result(&mut program)?;
        Ok(Self { inner: program, _marker: PhantomData })
    }

    pub fn load(fdl: &'a mut FileDataLoader) -> Result<Self> {
        let mut program = ffi::program_load(fdl.inner.as_mut().unwrap());
        let program = to_result(&mut program)?;
        Ok(Self { inner: program, _marker: PhantomData })
    }

    pub fn method(&self, name: &str, mgr: &mut MemoryManager) -> Result<Method> {
        cxx::let_cxx_string!(name = name);
        let mut method = ffi::program_load_method(&self.inner, &name, mgr.inner.as_mut().unwrap());
        let method = to_result(&mut method)?;
        Ok(Method { inner: method, _marker: PhantomData })
    }

    pub fn method_meta(&self, name: &str) -> Result<MethodMeta> {
        cxx::let_cxx_string!(name = name);
        let mut method_meta = ffi::program_method_meta(&self.inner, &name);
        let method_meta = to_result(&mut method_meta)?;
        Ok(MethodMeta { inner: method_meta, _marker: PhantomData })
    }
}

pub struct MethodMeta<'a> {
    inner: cxx::UniquePtr<ffi::MethodMeta>,
    _marker: PhantomData<&'a ()>,
}

impl MethodMeta<'_> {
    pub fn memory_manager(&self) -> MemoryManager {
        let mgr = ffi::program_memory_manager_for_method(&self.inner);
        MemoryManager { inner: mgr }
    }
}

pub struct MethodD<'a> {
    inner: cxx::UniquePtr<ffi::Method>,
    et_dump: ffi::ETDumpGen,
    _marker: PhantomData<&'a ()>,
}

pub struct Method<'a> {
    inner: cxx::UniquePtr<ffi::Method>,
    _marker: PhantomData<&'a ()>,
}

impl Method<'_> {
    pub fn inputs_size(&self) -> usize {
        self.inner.inputs_size()
    }

    pub fn outputs_size(&self) -> usize {
        self.inner.outputs_size()
    }

    /// # Safety
    ///
    /// The inputs that have been added via `set_input` must be still alive.
    pub unsafe fn execute(&mut self) -> Result<()> {
        let err = ffi::method_execute(self.inner.as_mut().unwrap());
        crate::error::from_ffi_err(err)?;
        Ok(())
    }

    pub fn set_input(&mut self, evalue: &EValue, idx: usize) -> Result<()> {
        let err = ffi::method_set_input(self.inner.as_mut().unwrap(), &evalue.inner, idx);
        crate::error::from_ffi_err(err)?;
        Ok(())
    }

    pub fn get_output(&self, idx: usize) -> EValueRef<'_> {
        let evalue = self.inner.get_output(idx);
        EValueRef { inner: evalue }
    }
}

impl MethodD<'_> {
    pub fn inputs_size(&self) -> usize {
        self.inner.inputs_size()
    }

    pub fn outputs_size(&self) -> usize {
        self.inner.outputs_size()
    }

    /// # Safety
    ///
    /// The inputs that have been added via `set_input` must be still alive.
    pub unsafe fn execute(&mut self) -> Result<()> {
        let err = ffi::method_execute(self.inner.as_mut().unwrap());
        crate::error::from_ffi_err(err)?;
        Ok(())
    }

    pub fn set_input(&mut self, evalue: &EValue, idx: usize) -> Result<()> {
        let err = ffi::method_set_input(self.inner.as_mut().unwrap(), &evalue.inner, idx);
        crate::error::from_ffi_err(err)?;
        Ok(())
    }

    pub fn get_output(&self, idx: usize) -> EValueRef<'_> {
        let evalue = self.inner.get_output(idx);
        EValueRef { inner: evalue }
    }
}

pub struct MemoryManager {
    inner: cxx::UniquePtr<ffi::MemoryManager>,
}

pub struct TensorImpl<'a> {
    inner: cxx::UniquePtr<ffi::TensorImpl>,
    #[allow(unused)]
    dims: Vec<i32>,
    _marker: PhantomData<&'a ()>,
}

impl TensorImpl<'_> {
    pub fn from_data<T: crate::scalar_type::WithScalarType>(data: &mut [T]) -> Self {
        let mut dims = vec![data.len() as i32];
        let tensor_impl = unsafe {
            ffi::tensor_impl(T::ST.c_int(), 1, dims.as_mut_ptr(), data.as_mut_ptr() as *mut u8)
        };
        Self { inner: tensor_impl, dims, _marker: PhantomData }
    }

    pub fn from_data_with_dims<T: crate::scalar_type::WithScalarType>(
        data: &mut [T],
        dims: &[usize],
    ) -> Result<Self> {
        let numel = dims.iter().product::<usize>();
        if numel != data.len() {
            crate::bail!("unexpected number of elements {} for dims {dims:?}", data.len())
        }
        let mut dims = dims.iter().map(|v| *v as i32).collect::<Vec<_>>();
        let tensor_impl = unsafe {
            ffi::tensor_impl(
                T::ST.c_int(),
                dims.len() as u32,
                dims.as_mut_ptr(),
                data.as_mut_ptr() as *mut u8,
            )
        };
        Ok(Self { inner: tensor_impl, dims, _marker: PhantomData })
    }
}

pub struct Tensor<'a> {
    inner: cxx::UniquePtr<ffi::Tensor>,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Tensor<'a> {
    pub fn new(impl_: &'a mut TensorImpl) -> Self {
        let tensor = ffi::tensor_new(impl_.inner.as_mut().unwrap());
        Self { inner: tensor, _marker: PhantomData }
    }

    pub fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel() as usize
    }

    pub fn dim(&self) -> usize {
        self.inner.dim() as usize
    }

    pub fn shape(&self) -> Vec<usize> {
        (0..self.inner.dim()).map(|i| self.inner.size(i) as usize).collect()
    }

    pub fn scalar_type(&self) -> crate::ScalarType {
        let st = ffi::tensor_scalar_type(&self.inner);
        crate::ScalarType::from_c_int(st).unwrap()
    }

    pub fn as_slice<T: crate::WithScalarType>(&self) -> Option<&[T]> {
        if self.scalar_type() != T::ST {
            return None;
        }
        let ptr = self.inner.const_data_ptr();
        let numel = self.numel();
        let data = unsafe { std::slice::from_raw_parts(ptr as *const T, numel) };
        Some(data)
    }
}

pub struct EValue<'a> {
    inner: cxx::UniquePtr<ffi::EValue>,
    _marker: PhantomData<&'a ()>,
}

impl EValue<'_> {
    pub fn from_tensor(tensor: &mut Tensor) -> Self {
        let evalue = ffi::evalue_from_tensor(tensor.inner.as_mut().unwrap());
        Self { inner: evalue, _marker: PhantomData }
    }
}

pub struct EValueRef<'a> {
    inner: &'a ffi::EValue,
}

impl EValueRef<'_> {
    pub fn is_tensor(&self) -> bool {
        self.inner.isTensor()
    }

    pub fn is_int(&self) -> bool {
        self.inner.isInt()
    }

    pub fn is_none(&self) -> bool {
        self.inner.isNone()
    }

    pub fn is_double(&self) -> bool {
        self.inner.isDouble()
    }

    pub fn is_string(&self) -> bool {
        self.inner.isString()
    }

    pub fn is_int_list(&self) -> bool {
        self.inner.isIntList()
    }

    pub fn is_double_list(&self) -> bool {
        self.inner.isDoubleList()
    }

    pub fn is_tensor_list(&self) -> bool {
        self.inner.isTensorList()
    }

    pub fn is_scalar(&self) -> bool {
        self.inner.isScalar()
    }

    pub fn tag(&self) -> Tag {
        Tag::from_c_int(ffi::evalue_tag(self.inner)).unwrap()
    }

    pub fn as_evalue(&self) -> crate::EValue {
        use crate::EValue as E;
        match self.tag() {
            Tag::None => E::None,
            Tag::Tensor => {
                let tensor_ref = TensorRef { inner: ffi::evalue_to_tensor(self.inner) };
                E::Tensor(tensor_ref)
            }
            Tag::Double => E::Double(self.inner.toDouble()),
            Tag::Int => E::Int(self.inner.toInt()),
            Tag::Bool => E::Bool(self.inner.toBool()),
            Tag::String => {
                let len = ffi::evalue_str_len(self.inner);
                let ptr = ffi::evalue_str_ptr(self.inner);
                let s = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec();
                E::String(s)
            }
            tag => E::Unsupported(tag),
        }
    }

    pub fn to_string(&self) -> Option<Vec<u8>> {
        if !self.is_string() {
            return None;
        }
        let len = ffi::evalue_str_len(self.inner);
        let ptr = ffi::evalue_str_ptr(self.inner);
        let s = unsafe { std::slice::from_raw_parts(ptr as *const u8, len) }.to_vec();
        Some(s)
    }

    pub fn as_tensor(&self) -> Option<TensorRef> {
        if !self.is_tensor() {
            return None;
        }
        Some(TensorRef { inner: ffi::evalue_to_tensor(self.inner) })
    }
}

pub struct TensorRef<'a> {
    inner: &'a ffi::Tensor,
}

impl TensorRef<'_> {
    pub fn const_data_ptr(&self) -> *const ffi::c_void {
        self.inner.const_data_ptr()
    }

    pub fn nbytes(&self) -> usize {
        self.inner.nbytes()
    }

    pub fn numel(&self) -> usize {
        self.inner.numel() as usize
    }

    pub fn dim(&self) -> usize {
        self.inner.dim() as usize
    }

    pub fn shape(&self) -> Vec<usize> {
        (0..self.inner.dim()).map(|i| self.inner.size(i) as usize).collect()
    }

    pub fn scalar_type(&self) -> crate::ScalarType {
        let st = ffi::tensor_scalar_type(self.inner);
        crate::ScalarType::from_c_int(st).unwrap()
    }

    pub fn as_slice<T: crate::WithScalarType>(&self) -> Option<&[T]> {
        if self.scalar_type() != T::ST {
            return None;
        }
        let ptr = self.inner.const_data_ptr();
        let numel = self.numel();
        // TODO: This expects the pointer to be on the host, check that it's always
        // the case.
        let data = unsafe { std::slice::from_raw_parts(ptr as *const T, numel) };
        Some(data)
    }
}
