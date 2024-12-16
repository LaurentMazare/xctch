use xctch_sys::safe;

#[ouroboros::self_referencing]
pub struct TensorImpl<T: crate::WithScalarType> {
    data: Vec<T>,
    #[borrows(mut data)]
    #[covariant]
    imp: safe::TensorImpl<'this>,
}

// Trying to have all three fields data, imp, and tensor, in the same struct doesn't seem
// to work with ouroboros 0.18 (but was fine with 0.17) so we split data and imp in the
// TensorImpl type.
// See https://github.com/someguynamedjosh/ouroboros/issues/100
#[ouroboros::self_referencing]
pub struct Tensor<T: crate::WithScalarType> {
    imp: TensorImpl<T>,
    #[borrows(mut imp)]
    #[covariant]
    pub tensor: safe::Tensor<'this>,
}

impl<T: crate::WithScalarType> Tensor<T> {
    pub fn from_data(data: Vec<T>) -> Self {
        let imp = TensorImplBuilder {
            data,
            imp_builder: |v: &mut Vec<T>| safe::TensorImpl::from_data(v.as_mut_slice()),
        }
        .build();
        TensorBuilder {
            imp,
            tensor_builder: |v: &mut TensorImpl<T>| v.with_imp_mut(safe::Tensor::new),
        }
        .build()
    }

    pub fn from_data_with_dims(data: Vec<T>, dims: &[usize]) -> crate::Result<Self> {
        let imp = TensorImplTryBuilder {
            data,
            imp_builder: |v: &mut Vec<T>| {
                safe::TensorImpl::from_data_with_dims(v.as_mut_slice(), dims)
            },
        }
        .try_build()?;
        let t = TensorBuilder {
            imp,
            tensor_builder: |v: &mut TensorImpl<T>| v.with_imp_mut(safe::Tensor::new),
        }
        .build();
        Ok(t)
    }

    pub fn nbytes(&self) -> usize {
        self.with_tensor(|v| v.nbytes())
    }

    pub fn as_evalue(&mut self) -> crate::EValue {
        self.with_tensor_mut(safe::EValue::from_tensor)
    }
}
