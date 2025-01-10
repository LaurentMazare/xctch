use crate::Result;
use xctch_sys::safe;

#[ouroboros::self_referencing]
struct BufferDataLoader {
    buffer: Vec<u8>,
    #[borrows(buffer)]
    #[not_covariant]
    inner: xctch_sys::safe::BufferDataLoader<'this>,
}

impl BufferDataLoader {
    pub fn from_buffer(buffer: Vec<u8>) -> Self {
        BufferDataLoaderBuilder { buffer, inner_builder: |buf| safe::BufferDataLoader::new(buf) }
            .build()
    }
}

#[ouroboros::self_referencing]
pub struct ProgramBuffer {
    bdl: BufferDataLoader,
    #[borrows(mut bdl)]
    #[not_covariant]
    inner: xctch_sys::safe::Program<'this>,
}

impl ProgramBuffer {
    pub fn from_buffer(buffer: Vec<u8>) -> Result<Self> {
        let bdl = BufferDataLoader::from_buffer(buffer);
        ProgramBufferTryBuilder {
            bdl,
            inner_builder: |bdl| bdl.with_inner_mut(safe::Program::load_b),
        }
        .try_build()
    }

    pub fn method(&self, name: &str) -> Result<crate::Method> {
        let meta = self.with_inner(|v| v.method_meta(name))?;
        let mut mgr = meta.memory_manager();
        let method = self.with_inner(|v| v.method(name, &mut mgr))?;
        Ok(crate::Method { method, meta, mgr })
    }

    pub fn method_d(&self, name: &str) -> Result<crate::MethodD> {
        let meta = self.with_inner(|v| v.method_meta(name))?;
        let mut mgr = meta.memory_manager();
        let method = self.with_inner(|v| v.method_d(name, &mut mgr))?;
        Ok(crate::MethodD { method, meta, mgr })
    }
}

#[ouroboros::self_referencing]
pub struct ProgramFile {
    fdl: safe::FileDataLoader,
    #[borrows(mut fdl)]
    #[not_covariant]
    inner: xctch_sys::safe::Program<'this>,
}

impl ProgramFile {
    pub fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let p = p.as_ref();
        let fdl = safe::FileDataLoader::new(p).map_err(|v| v.with_path(p))?;
        ProgramFileTryBuilder {
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

    pub fn method_d(&self, name: &str) -> Result<crate::MethodD> {
        let meta = self.with_inner(|v| v.method_meta(name))?;
        let mut mgr = meta.memory_manager();
        let method = self.with_inner(|v| v.method_d(name, &mut mgr))?;
        Ok(crate::MethodD { method, meta, mgr })
    }
}

pub enum Program {
    File(ProgramFile),
    Buffer(ProgramBuffer),
}

impl Program {
    pub fn from_file<P: AsRef<std::path::Path>>(p: P) -> Result<Self> {
        let program = ProgramFile::from_file(p)?;
        Ok(Self::File(program))
    }

    pub fn from_buffer(buffer: Vec<u8>) -> Result<Self> {
        let program = ProgramBuffer::from_buffer(buffer)?;
        Ok(Self::Buffer(program))
    }

    pub fn method(&self, name: &str) -> Result<crate::Method> {
        match self {
            Self::Buffer(s) => s.method(name),
            Self::File(s) => s.method(name),
        }
    }

    pub fn method_d(&self, name: &str) -> Result<crate::MethodD> {
        match self {
            Self::Buffer(s) => s.method_d(name),
            Self::File(s) => s.method_d(name),
        }
    }
}
