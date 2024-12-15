#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum ScalarType {
    U8,
    I8,
    I16,
    I32,
    I64,
    F16,
    F32,
    F64,
    Bool,
    BF16,
}

impl ScalarType {
    pub fn c_int(&self) -> libc::c_int {
        match self {
            Self::U8 => 0,
            Self::I8 => 1,
            Self::I16 => 2,
            Self::I32 => 3,
            Self::I64 => 4,
            Self::F16 => 5,
            Self::F32 => 6,
            Self::F64 => 7,
            Self::Bool => 11,
            Self::BF16 => 15,
        }
    }

    pub fn from_c_int(v: libc::c_int) -> Option<Self> {
        match v {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::I16),
            3 => Some(Self::I32),
            4 => Some(Self::I64),
            5 => Some(Self::F16),
            6 => Some(Self::F32),
            7 => Some(Self::F64),
            11 => Some(Self::Bool),
            15 => Some(Self::BF16),
            _ => None,
        }
    }
}

pub trait WithScalarType {
    const ST: ScalarType;
}

impl WithScalarType for u8 {
    const ST: ScalarType = ScalarType::U8;
}
impl WithScalarType for i8 {
    const ST: ScalarType = ScalarType::I8;
}
impl WithScalarType for i16 {
    const ST: ScalarType = ScalarType::I16;
}
impl WithScalarType for i32 {
    const ST: ScalarType = ScalarType::I32;
}
impl WithScalarType for i64 {
    const ST: ScalarType = ScalarType::I64;
}
impl WithScalarType for half::f16 {
    const ST: ScalarType = ScalarType::F16;
}
impl WithScalarType for f32 {
    const ST: ScalarType = ScalarType::F32;
}
impl WithScalarType for f64 {
    const ST: ScalarType = ScalarType::F64;
}
impl WithScalarType for bool {
    const ST: ScalarType = ScalarType::Bool;
}
impl WithScalarType for half::bf16 {
    const ST: ScalarType = ScalarType::BF16;
}
