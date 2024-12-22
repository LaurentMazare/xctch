#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum FfiError {
    /// Status indicating a successful operation.
    Ok = 0x00,

    /// An internal error occurred.
    Internal = 0x01,

    /// Status indicating the executor is in an invalid state for a target
    /// operation
    InvalidState = 0x2,

    /// Status indicating there are no more steps of execution to run
    EndOfMethod = 0x03,

    /*
     * Logical errors.
     */
    /// Operation is not supported in the current context.
    NotSupported = 0x10,

    /// Operation is not yet implemented.
    NotImplemented = 0x11,

    /// User provided an invalid argument.
    InvalidArgument = 0x12,

    /// Object is an invalid type for the operation.
    InvalidType = 0x13,

    /// Operator(s) missing in the operator registry.
    OperatorMissing = 0x14,

    /*
     * Resource errors.
     */
    /// Requested resource could not be found.
    NotFound = 0x20,

    /// Could not allocate the requested memory.
    MemoryAllocationFailed = 0x21,

    /// Could not access a resource.
    AccessFailed = 0x22,

    /// Error caused by the contents of a program.
    InvalidProgram = 0x23,

    /*
     * Delegate errors.
     */
    /// Init stage: Backend receives an incompatible delegate version.
    DelegateInvalidCompatibility = 0x30,
    /// Init stage: Backend fails to allocate memory.
    DelegateMemoryAllocationFailed = 0x31,
    /// Execute stage: The handle is invalid.
    DelegateInvalidHandle = 0x32,
}

impl FfiError {
    pub fn from_u32(value: u32) -> Option<Self> {
        match value {
            0x00 => Some(Self::Ok),
            0x01 => Some(Self::Internal),
            0x02 => Some(Self::InvalidState),
            0x03 => Some(Self::EndOfMethod),
            0x10 => Some(Self::NotSupported),
            0x11 => Some(Self::NotImplemented),
            0x12 => Some(Self::InvalidArgument),
            0x13 => Some(Self::InvalidType),
            0x14 => Some(Self::OperatorMissing),
            0x20 => Some(Self::NotFound),
            0x21 => Some(Self::MemoryAllocationFailed),
            0x22 => Some(Self::AccessFailed),
            0x23 => Some(Self::InvalidProgram),
            0x30 => Some(Self::DelegateInvalidCompatibility),
            0x31 => Some(Self::DelegateMemoryAllocationFailed),
            0x32 => Some(Self::DelegateInvalidHandle),
            _ => None,
        }
    }
}

impl std::fmt::Display for FfiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for FfiError {}

impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

/// Main library error type.
#[derive(thiserror::Error)]
pub enum Error {
    /// Integer parse error.
    #[error(transparent)]
    ParseInt(#[from] std::num::ParseIntError),

    /// I/O error.
    #[error(transparent)]
    Io(#[from] std::io::Error),

    #[error("null ffi pointer")]
    NullFfiPtr,

    #[error(transparent)]
    FromUtf8(#[from] std::string::FromUtf8Error),

    #[error(transparent)]
    BorrowMutError(#[from] std::cell::BorrowMutError),

    #[error(transparent)]
    BorrowError(#[from] std::cell::BorrowError),

    /// Arbitrary errors wrapping.
    #[error("{0}")]
    Wrapped(Box<dyn std::fmt::Display + Send + Sync>),

    #[error("{context}\n{inner}")]
    Context { inner: Box<Self>, context: Box<dyn std::fmt::Display + Send + Sync> },

    /// Adding path information to an error.
    #[error("path: {path:?} {inner}")]
    WithPath { inner: Box<Self>, path: std::path::PathBuf },

    #[error("{inner}\n{backtrace}")]
    WithBacktrace { inner: Box<Self>, backtrace: Box<std::backtrace::Backtrace> },

    /// Ffi Error
    #[error(transparent)]
    FfiError(#[from] FfiError),

    /// User generated error message, typically created via `bail!`.
    #[error("{0}")]
    Msg(String),

    #[error("cannot find tensor {path}")]
    CannotFindTensor { path: String },

    #[error("unwrap none")]
    UnwrapNone,
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    pub fn wrap(err: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Wrapped(Box::new(err)).bt()
    }

    pub fn msg(err: impl std::error::Error) -> Self {
        Self::Msg(err.to_string()).bt()
    }

    pub fn debug(err: impl std::fmt::Debug) -> Self {
        Self::Msg(format!("{err:?}")).bt()
    }

    pub fn bt(self) -> Self {
        let backtrace = std::backtrace::Backtrace::capture();
        match backtrace.status() {
            std::backtrace::BacktraceStatus::Disabled
            | std::backtrace::BacktraceStatus::Unsupported => self,
            _ => Self::WithBacktrace { inner: Box::new(self), backtrace: Box::new(backtrace) },
        }
    }

    pub fn with_path<P: AsRef<std::path::Path>>(self, p: P) -> Self {
        Self::WithPath { inner: Box::new(self), path: p.as_ref().to_path_buf() }
    }

    pub fn context(self, c: impl std::fmt::Display + Send + Sync + 'static) -> Self {
        Self::Context { inner: Box::new(self), context: Box::new(c) }
    }
}

impl<T> From<std::sync::PoisonError<T>> for Error {
    fn from(_value: std::sync::PoisonError<T>) -> Self {
        Self::Msg("poisoned mutex".into())
    }
}

#[macro_export]
macro_rules! bail {
    ($msg:literal $(,)?) => {
        return Err($crate::Error::Msg(format!($msg).into()).bt())
    };
    ($err:expr $(,)?) => {
        return Err($crate::Error::Msg(format!($err).into()).bt())
    };
    ($fmt:expr, $($arg:tt)*) => {
        return Err($crate::Error::Msg(format!($fmt, $($arg)*).into()).bt())
    };
}

pub(crate) fn from_ffi_err(err: u32) -> Result<()> {
    let err = match FfiError::from_u32(err) {
        Some(err) => err,
        None => Err(Error::FfiError(FfiError::Internal).bt())?,
    };
    if err != FfiError::Ok {
        Err(Error::FfiError(err).bt())?
    }
    Ok(())
}

// Taken from anyhow.
pub trait Context<T> {
    /// Wrap the error value with additional context.
    fn context<C>(self, context: C) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static;

    /// Wrap the error value with additional context that is evaluated lazily
    /// only once an error does occur.
    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C;
}

impl<T> Context<T> for Option<T> {
    fn context<C>(self, context: C) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::UnwrapNone.context(context).bt()),
        }
    }

    fn with_context<C, F>(self, f: F) -> Result<T>
    where
        C: std::fmt::Display + Send + Sync + 'static,
        F: FnOnce() -> C,
    {
        match self {
            Some(v) => Ok(v),
            None => Err(Error::UnwrapNone.context(f()).bt()),
        }
    }
}
