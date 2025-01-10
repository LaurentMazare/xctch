use crate::Result;
use xctch_sys::safe;

#[allow(unused)]
pub struct Method<'a> {
    pub(crate) method: xctch_sys::safe::Method<'a>,
    pub(crate) meta: xctch_sys::safe::MethodMeta<'a>,
    pub(crate) mgr: xctch_sys::safe::MemoryManager,
}

impl Method<'_> {
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

#[allow(unused)]
pub struct MethodD<'a> {
    pub(crate) method: xctch_sys::safe::MethodD<'a>,
    pub(crate) meta: xctch_sys::safe::MethodMeta<'a>,
    pub(crate) mgr: xctch_sys::safe::MemoryManager,
}

impl MethodD<'_> {
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

    pub fn dump_data(&mut self) -> Vec<u8> {
        self.method.dump_data()
    }
}
