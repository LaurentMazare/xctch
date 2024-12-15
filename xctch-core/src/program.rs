use crate::Result;
use xctch_sys::safe;

pub struct Program {
    inner: xctch_sys::safe::Program,
}

impl Program {
    pub fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let mut fdl = safe::FileDataLoader::new(p).map_err(|v| v.with_path(p))?;
        let program = safe::Program::load(&mut fdl).map_err(|v| v.with_path(p))?;
        // TODO: Fix the memory leak but still ensure that fdl is kept alive.
        Box::leak(Box::new(fdl));
        Ok(Self { inner: program })
    }

    pub fn method(&self, name: &str) -> Result<crate::Method> {
        let meta = self.inner.method_meta(name)?;
        let mut mgr = meta.memory_manager();
        let method = self.inner.method(name, &mut mgr)?;
        Ok(crate::Method { method, meta, mgr })
    }
}
