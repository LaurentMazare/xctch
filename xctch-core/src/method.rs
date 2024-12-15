use crate::Result;
use xctch_sys::safe;

#[allow(unused)]
pub struct Method {
    pub(crate) method: xctch_sys::safe::Method,
    pub(crate) meta: xctch_sys::safe::MethodMeta,
    pub(crate) mgr: xctch_sys::safe::MemoryManager,
}

impl Method {
    pub fn set_input(&mut self, input: &safe::EValue<'_>, idx: usize) -> Result<()> {
        self.method.set_input(input, idx)
    }

    /// # Safety
    ///
    /// The inputs that have been added via `set_input` must be still alive.
    pub unsafe fn execute(&mut self) -> Result<()> {
        self.method.execute()
    }

    pub fn get_output(&self, idx: usize) -> safe::EValueRef<'_> {
        self.method.get_output(idx)
    }

    pub fn inputs_size(&self) -> usize {
        self.method.inputs_size()
    }

    pub fn outputs_size(&self) -> usize {
        self.method.outputs_size()
    }
}
