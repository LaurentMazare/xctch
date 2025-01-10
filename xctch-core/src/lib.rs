pub mod method;
pub mod program;
pub mod tensor;

pub use method::{Method, MethodD};
pub use program::{Program, ProgramBuffer, ProgramFile};
pub use scalar_type::{ScalarType, WithScalarType};
pub use tensor::Tensor;
pub use xctch_sys::safe::{EValue, EValueRef, TensorRef};
pub use xctch_sys::{bail, scalar_type, Context, Error, Result};

pub fn et_pal_init() {
    unsafe { xctch_sys::et_pal_init() }
}
