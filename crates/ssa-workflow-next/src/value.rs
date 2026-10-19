/// Value serialization owned by the workflow. Checksums and compression belong to the store.
pub trait Value: Sized + Send + Sync {
    fn encode(&self, out: &mut Vec<u8>);
    fn decode(bytes: &[u8]) -> Result<Self, String>;
}

macro_rules! scalar_value {
    ($($ty:ty),*) => {$(
        impl Value for $ty {
            fn encode(&self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }

            fn decode(bytes: &[u8]) -> Result<Self, String> {
                let bytes = bytes.try_into().map_err(|_| format!("expected {} bytes", size_of::<$ty>()))?;
                Ok(<$ty>::from_le_bytes(bytes))
            }
        }

        impl Value for Vec<$ty> {
            fn encode(&self, out: &mut Vec<u8>) {
                out.reserve(self.len() * size_of::<$ty>());
                for value in self {
                    out.extend_from_slice(&value.to_le_bytes());
                }
            }

            fn decode(bytes: &[u8]) -> Result<Self, String> {
                if bytes.len() % size_of::<$ty>() != 0 {
                    return Err(format!("length {} is not a multiple of {}", bytes.len(), size_of::<$ty>()));
                }
                Ok(bytes
                    .chunks_exact(size_of::<$ty>())
                    .map(|chunk| <$ty>::from_le_bytes(chunk.try_into().expect("chunks_exact yields exact chunks")))
                    .collect())
            }
        }
    )*};
}

scalar_value!(u32, u64, f64);
