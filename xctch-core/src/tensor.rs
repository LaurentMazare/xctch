use xctch_sys::safe;

// This requires ouroboros 0.17 and is not compatible with 0.18.
// See https://github.com/someguynamedjosh/ouroboros/issues/100
#[ouroboros::self_referencing]
pub struct Tensor<T: crate::WithScalarType> {
    data: Vec<T>,
    #[borrows(mut data)]
    #[not_covariant]
    imp: safe::TensorImpl<'this>,
    #[borrows(mut imp)]
    #[not_covariant]
    pub tensor: safe::Tensor<'this>,
}

impl<T: crate::WithScalarType> Tensor<T> {
    pub fn from_data(data: Vec<T>) -> Self {
        TensorBuilder {
            data,
            imp_builder: |v: &mut Vec<T>| safe::TensorImpl::from_data(v.as_mut_slice()),
            tensor_builder: |v: &mut safe::TensorImpl| safe::Tensor::new(v),
        }
        .build()
    }

    pub fn nbytes(&self) -> usize {
        self.with_tensor(|v| v.nbytes())
    }
}
