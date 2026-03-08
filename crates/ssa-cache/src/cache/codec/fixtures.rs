use crate::{
    Result,
    cache::codec::{CodecEngine, SkipReason},
    error::Error as GlobalError,
};

#[derive(Debug, Default)]
pub(crate) struct FixtureEngine {
    buffer: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FixtureTag(u8);

impl FixtureTag {
    const CONTAINER_BITS: u8 = 3;
    const CONTAINER_SHIFT: u8 = 8 - Self::CONTAINER_BITS;

    const fn new(container: ContainerKind, element: ElementKind) -> Self {
        Self(((container as u8) << Self::CONTAINER_SHIFT) | (element as u8))
    }

    const fn into_inner(self) -> u8 {
        self.0
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ContainerKind {
    Scalar = 0,
    String = 1,
    Vec = 2,
    Array = 3,
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ElementKind {
    None = 0,
    U8 = 1,
    U16 = 2,
    U32 = 3,
    U64 = 4,
    Usize = 5,
    I8 = 6,
    I16 = 7,
    I32 = 8,
    I64 = 9,
    Isize = 10,
    F32 = 11,
    F64 = 12,
}

pub(crate) trait FixtureValue: Sized {
    const TAG: FixtureTag;

    fn encode_to(&self, output: &mut Vec<u8>);
    fn decode_from(bytes: &[u8]) -> Result<Self>;
}

trait FixtureScalar: Sized {
    const ELEMENT: ElementKind;
    const WIDTH: usize = core::mem::size_of::<Self>();

    fn encode_scalar(&self, output: &mut Vec<u8>);
    fn decode_scalar(bytes: &[u8]) -> Result<Self>;
}

impl<T: FixtureValue> CodecEngine<T> for FixtureEngine {
    fn encode(&mut self, value: &T) -> Result<&[u8], SkipReason> {
        self.buffer.clear();
        self.buffer.push(T::TAG.into_inner());
        value.encode_to(&mut self.buffer);
        Ok(&self.buffer)
    }

    fn decode(&mut self, bytes: &[u8]) -> Result<T> {
        let Some((&tag, payload)) = bytes.split_first() else {
            return Err(fixture_error("missing fixture tag"));
        };
        if tag != T::TAG.into_inner() {
            return Err(fixture_error("fixture type tag mismatch"));
        }
        T::decode_from(payload)
    }
}

impl<T: FixtureScalar> FixtureValue for T {
    const TAG: FixtureTag = FixtureTag::new(ContainerKind::Scalar, T::ELEMENT);

    fn encode_to(&self, output: &mut Vec<u8>) {
        self.encode_scalar(output);
    }

    fn decode_from(bytes: &[u8]) -> Result<Self> {
        T::decode_scalar(bytes)
    }
}

impl<T: FixtureScalar> FixtureValue for Vec<T> {
    const TAG: FixtureTag = FixtureTag::new(ContainerKind::Vec, T::ELEMENT);

    fn encode_to(&self, output: &mut Vec<u8>) {
        let len = u32::try_from(self.len()).expect("fixture vec length must fit into u32");
        output.extend_from_slice(&len.to_le_bytes());
        for value in self {
            value.encode_scalar(output);
        }
    }

    fn decode_from(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < core::mem::size_of::<u32>() {
            return Err(fixture_error("fixture vec payload is truncated"));
        }

        let (len_bytes, payload) = bytes.split_at(core::mem::size_of::<u32>());
        let len = u32::from_le_bytes(
            len_bytes
                .try_into()
                .expect("fixture vec length prefix is exactly 4 bytes"),
        ) as usize;
        let expected = len
            .checked_mul(T::WIDTH)
            .ok_or_else(|| fixture_error("fixture vec payload length overflows"))?;
        if payload.len() != expected {
            return Err(fixture_error("fixture vec payload length mismatch"));
        }

        payload
            .chunks_exact(T::WIDTH)
            .map(T::decode_scalar)
            .collect()
    }
}

impl<T: FixtureScalar, const N: usize> FixtureValue for [T; N] {
    const TAG: FixtureTag = FixtureTag::new(ContainerKind::Array, T::ELEMENT);

    fn encode_to(&self, output: &mut Vec<u8>) {
        for value in self {
            value.encode_scalar(output);
        }
    }

    fn decode_from(bytes: &[u8]) -> Result<Self> {
        let expected = N
            .checked_mul(T::WIDTH)
            .ok_or_else(|| fixture_error("fixture array payload length overflows"))?;
        if bytes.len() != expected {
            return Err(fixture_error("fixture array payload length mismatch"));
        }

        let values: Vec<T> = bytes
            .chunks_exact(T::WIDTH)
            .map(T::decode_scalar)
            .collect::<Result<_>>()?;
        values
            .try_into()
            .map_err(|_| fixture_error("fixture array element count mismatch"))
    }
}

impl FixtureValue for String {
    const TAG: FixtureTag = FixtureTag::new(ContainerKind::String, ElementKind::None);

    fn encode_to(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(self.as_bytes());
    }

    fn decode_from(bytes: &[u8]) -> Result<Self> {
        String::from_utf8(bytes.to_vec()).map_err(|_| fixture_error("fixture string is not utf-8"))
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("{0}")]
    Invalid(&'static str),
}

fn fixture_error(message: &'static str) -> GlobalError {
    GlobalError::from(super::Error::from(Error::Invalid(message)))
}

fn decode_fixed<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| fixture_error("fixture payload length mismatch"))
}

macro_rules! impl_fixture_scalar {
    ($ty:ty, $element:expr) => {
        impl FixtureScalar for $ty {
            const ELEMENT: ElementKind = $element;

            fn encode_scalar(&self, output: &mut Vec<u8>) {
                output.extend_from_slice(&self.to_le_bytes());
            }

            fn decode_scalar(bytes: &[u8]) -> Result<Self> {
                Ok(Self::from_le_bytes(decode_fixed(bytes)?))
            }
        }
    };
}

impl_fixture_scalar!(u8, ElementKind::U8);
impl_fixture_scalar!(u16, ElementKind::U16);
impl_fixture_scalar!(u32, ElementKind::U32);
impl_fixture_scalar!(u64, ElementKind::U64);
impl_fixture_scalar!(usize, ElementKind::Usize);
impl_fixture_scalar!(i8, ElementKind::I8);
impl_fixture_scalar!(i16, ElementKind::I16);
impl_fixture_scalar!(i32, ElementKind::I32);
impl_fixture_scalar!(i64, ElementKind::I64);
impl_fixture_scalar!(isize, ElementKind::Isize);
impl_fixture_scalar!(f32, ElementKind::F32);
impl_fixture_scalar!(f64, ElementKind::F64);
