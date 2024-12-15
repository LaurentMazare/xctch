pub mod method;
pub mod program;
pub mod tensor;

pub use method::Method;
pub use program::Program;
pub use tensor::Tensor;
pub use xctch_sys::safe::EValue;
pub use xctch_sys::scalar_type;
pub use xctch_sys::{scalar_type::ScalarType, scalar_type::WithScalarType, Error, Result};

pub fn et_pal_init() {
    unsafe { xctch_sys::et_pal_init() }
}
