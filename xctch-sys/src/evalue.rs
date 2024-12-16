#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum Tag {
    None,
    Tensor,
    String,
    Double,
    Int,
    Bool,
    ListBool,
    ListDouble,
    ListInt,
    ListTensor,
    ListScalar,
    ListOptionalTensor,
}

impl Tag {
    pub fn c_int(&self) -> u32 {
        match self {
            Self::None => 0,
            Self::Tensor => 1,
            Self::String => 2,
            Self::Double => 3,
            Self::Int => 4,
            Self::Bool => 5,
            Self::ListBool => 6,
            Self::ListDouble => 7,
            Self::ListInt => 8,
            Self::ListTensor => 9,
            Self::ListScalar => 10,
            Self::ListOptionalTensor => 11,
        }
    }

    pub fn from_c_int(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Tensor),
            2 => Some(Self::String),
            3 => Some(Self::Double),
            4 => Some(Self::Int),
            5 => Some(Self::Bool),
            6 => Some(Self::ListBool),
            7 => Some(Self::ListDouble),
            8 => Some(Self::ListInt),
            9 => Some(Self::ListTensor),
            10 => Some(Self::ListScalar),
            11 => Some(Self::ListOptionalTensor),
            _ => None,
        }
    }
}

pub enum EValue<'a> {
    None,
    Tensor(crate::safe::TensorRef<'a>),
    String(Vec<u8>),
    Double(f64),
    Int(i64),
    Bool(bool),
    Unsupported(Tag),
}
