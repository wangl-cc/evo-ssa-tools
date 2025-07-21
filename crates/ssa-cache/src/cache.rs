use crate::Result;

pub trait CodecBuffer {
    fn init() -> Self;
}

pub trait Cacheable: Sized {
    type Buffer: CodecBuffer;

    fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8];

    fn decode(bytes: &[u8], buffer: &mut Self::Buffer) -> Result<Self>;
}

#[cfg(feature = "bitcode")]
impl CodecBuffer for bitcode::Buffer {
    fn init() -> Self {
        Self::new()
    }
}

#[cfg(feature = "bitcode")]
impl<T> Cacheable for T
where
    T: bitcode::Encode + for<'b> bitcode::Decode<'b>,
{
    type Buffer = bitcode::Buffer;

    fn encode<'b>(&self, buffer: &'b mut Self::Buffer) -> &'b [u8] {
        buffer.encode(self)
    }

    fn decode(bytes: &[u8], buffer: &mut Self::Buffer) -> Result<Self> {
        Ok(buffer.decode(bytes)?)
    }
}

pub trait CacheStore<S: ?Sized>: Sync {
    fn fetch<T: Cacheable>(&self, sig: &S, buffer: &mut T::Buffer) -> Result<Option<T>>;

    fn store<T: Cacheable>(&self, sig: &S, buffer: &mut T::Buffer, value: &T) -> Result<()>;
}

#[cfg(feature = "fjall")]
mod fjall_store {
    use super::*;

    impl CacheStore<[u8]> for fjall::Partition {
        fn fetch<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer) -> Result<Option<T>> {
            let value = self.get(sig)?;
            value
                .map(|value| T::decode(value.as_ref(), buffer))
                .transpose()
        }

        fn store<T: Cacheable>(&self, sig: &[u8], buffer: &mut T::Buffer, value: &T) -> Result<()> {
            let value = value.encode(buffer);
            self.insert(sig, value)?;
            Ok(())
        }
    }
}
