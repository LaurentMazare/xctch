use crate::Result;
use xctch_sys::safe;

#[ouroboros::self_referencing]
pub struct Program {
    fdl: safe::FileDataLoader,
    #[borrows(mut fdl)]
    #[not_covariant]
    inner: xctch_sys::safe::Program<'this>,
}

impl Program {
    pub fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let fdl = safe::FileDataLoader::new(p).map_err(|v| v.with_path(p))?;
        ProgramTryBuilder {
            fdl,
            inner_builder: |fdl| safe::Program::load(fdl).map_err(|v| v.with_path(p)),
        }
        .try_build()
    }

    pub fn method(&self, name: &str) -> Result<crate::Method> {
        let meta = self.with_inner(|v| v.method_meta(name))?;
        let mut mgr = meta.memory_manager();
        let method = self.with_inner(|v| v.method(name, &mut mgr))?;
        Ok(crate::Method { method, meta, mgr })
    }
}
