use crate::Result;
use xctch_sys::safe;

pub struct Method {
    pub(crate) method: xctch_sys::safe::Method,
    pub(crate) meta: xctch_sys::safe::MethodMeta,
    pub(crate) mgr: xctch_sys::safe::MemoryManager,
}

impl Method {
    pub fn set_input(&mut self, input: &safe::EValue<'_>, idx: usize) -> Result<()> {
        self.method.set_input(input, idx)
    }

    pub unsafe fn execute(&mut self) -> Result<()> {
        self.method.execute()
    }

    pub fn get_output(&self, idx: usize) -> safe::EValueRef<'_> {
        self.method.get_output(idx)
    }
}
