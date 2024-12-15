use crate::Result;
use xctch_sys::safe;

pub struct Program {
    inner: xctch_sys::safe::Program,
}

impl Program {
    pub fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let mut fdl = safe::FileDataLoader::new(p)?;
        let program = safe::Program::load(&mut fdl)?;
        Ok(Self { inner: program })
    }

    pub fn method(&self, name: &str) -> Result<crate::Method> {
        let meta = self.inner.method_meta(name)?;
        let mut mgr = meta.memory_manager();
        let method = self.inner.method(name, &mut mgr)?;
        Ok(crate::Method { method, meta, mgr })
    }
}
