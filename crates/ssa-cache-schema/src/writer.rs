use crate::SchemaFingerprint;

const DOMAIN_VERSION: &[u8] = b"ssa-cache-schema:v1";

/// Canonical writer used by [`crate::CacheSchema`] implementations.
///
/// New writers are seeded with the fixed `ssa-cache-schema:v1` domain/version header. Every token
/// is encoded as a one-byte tag followed by fixed-width integers or length-prefixed byte strings.
/// This keeps different schema trees from colliding through ambiguous concatenation.
#[derive(Clone)]
pub struct SchemaWriter {
    hasher: blake3::Hasher,
}

// Construction.
impl SchemaWriter {
    /// Create a schema writer seeded with the schema domain/version header.
    pub fn new() -> Self {
        let mut hasher = blake3::Hasher::new();
        hasher.update(DOMAIN_VERSION);
        Self { hasher }
    }
}

// Type identity tokens.
impl SchemaWriter {
    /// Write a primitive type name.
    pub fn primitive(&mut self, name: &str) {
        self.tagged_str(Tag::Primitive, name);
    }

    /// Write an explicit type schema version salt.
    pub fn type_version(&mut self, version: &str) {
        self.tagged_str(Tag::TypeVersion, version);
    }
}

// Product type tokens.
impl SchemaWriter {
    /// Begin a struct schema.
    pub fn struct_begin(&mut self, name: &str) {
        self.tag(Tag::StructBegin);
        self.str(name);
    }

    /// End a struct schema.
    pub fn struct_end(&mut self) {
        self.tag(Tag::StructEnd);
    }

    /// Begin a field schema.
    pub fn field_begin(&mut self, index: usize, name: Option<&str>) {
        self.tag(Tag::FieldBegin);
        self.usize(index);
        self.option_str(name);
    }

    /// End a field schema.
    pub fn field_end(&mut self) {
        self.tag(Tag::FieldEnd);
    }
}

// Sum type tokens.
impl SchemaWriter {
    /// Begin an enum schema.
    pub fn enum_begin(&mut self, name: &str) {
        self.tag(Tag::EnumBegin);
        self.str(name);
    }

    /// End an enum schema.
    pub fn enum_end(&mut self) {
        self.tag(Tag::EnumEnd);
    }

    /// Begin an enum variant schema.
    pub fn variant_begin(&mut self, index: usize, name: &str) {
        self.tag(Tag::VariantBegin);
        self.usize(index);
        self.str(name);
    }

    /// End an enum variant schema.
    pub fn variant_end(&mut self) {
        self.tag(Tag::VariantEnd);
    }
}

// Tuple type tokens.
impl SchemaWriter {
    /// Begin a tuple schema.
    pub fn tuple_begin(&mut self) {
        self.tag(Tag::TupleBegin);
    }

    /// End a tuple schema.
    pub fn tuple_end(&mut self) {
        self.tag(Tag::TupleEnd);
    }
}

// Collection type tokens.
impl SchemaWriter {
    /// Begin a sequence-like schema.
    pub fn seq_begin(&mut self, name: &str) {
        self.tagged_str(Tag::SeqBegin, name);
    }

    /// End a sequence-like schema.
    pub fn seq_end(&mut self) {
        self.tag(Tag::SeqEnd);
    }

    /// Begin an array schema with fixed length.
    pub fn array_begin(&mut self, len: usize) {
        self.tag(Tag::ArrayBegin);
        self.usize(len);
    }

    /// End an array schema.
    pub fn array_end(&mut self) {
        self.tag(Tag::ArrayEnd);
    }

    /// Begin a map-like schema.
    pub fn map_begin(&mut self, name: &str) {
        self.tagged_str(Tag::MapBegin, name);
    }

    /// End a map-like schema.
    pub fn map_end(&mut self) {
        self.tag(Tag::MapEnd);
    }
}

// Crate-private finalization.
impl SchemaWriter {
    pub(crate) fn finish_fingerprint(self) -> SchemaFingerprint {
        let full = self.hasher.finalize();
        full.as_bytes()[..16]
            .try_into()
            .expect("BLAKE3 hash is at least 16 bytes")
    }
}

// Low-level canonical encoding.
impl SchemaWriter {
    fn tagged_str(&mut self, tag: Tag, value: &str) {
        self.tag(tag);
        self.str(value);
    }

    fn option_str(&mut self, value: Option<&str>) {
        match value {
            Some(value) => {
                self.tag(Tag::Some);
                self.str(value);
            }
            None => self.tag(Tag::None),
        }
    }

    fn str(&mut self, value: &str) {
        self.usize(value.len());
        self.write(value.as_bytes());
    }

    fn usize(&mut self, value: usize) {
        self.write(&(value as u64).to_le_bytes());
    }

    fn tag(&mut self, tag: Tag) {
        self.write(&[tag as u8]);
    }

    fn write(&mut self, bytes: &[u8]) {
        self.hasher.update(bytes);
    }
}

impl Default for SchemaWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for SchemaWriter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaWriter").finish_non_exhaustive()
    }
}

#[repr(u8)]
#[derive(Clone, Copy)]
enum Tag {
    Primitive = 1,
    StructBegin = 2,
    StructEnd = 3,
    EnumBegin = 4,
    EnumEnd = 5,
    TypeVersion = 6,
    FieldBegin = 7,
    FieldEnd = 8,
    VariantBegin = 9,
    VariantEnd = 10,
    TupleBegin = 11,
    TupleEnd = 12,
    SeqBegin = 13,
    SeqEnd = 14,
    MapBegin = 15,
    MapEnd = 16,
    Some = 17,
    None = 18,
    ArrayBegin = 19,
    ArrayEnd = 20,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn hash_to_fingerprint(hasher: blake3::Hasher) -> SchemaFingerprint {
        let full = hasher.finalize();
        full.as_bytes()[..16]
            .try_into()
            .expect("BLAKE3 hash is at least 16 bytes")
    }

    #[test]
    fn writer_seeds_fingerprint_with_domain_version() {
        let mut writer = SchemaWriter::new();
        writer.primitive("u32");
        let actual = writer.finish_fingerprint();

        let mut expected = blake3::Hasher::new();
        expected.update(DOMAIN_VERSION);
        expected.update(&[Tag::Primitive as u8]);
        expected.update(&3_u64.to_le_bytes());
        expected.update(b"u32");

        assert_eq!(actual, hash_to_fingerprint(expected));
    }

    #[test]
    fn writer_uses_length_prefixes_for_strings() {
        let mut first = SchemaWriter::new();
        first.seq_begin("ab");
        first.primitive("c");
        first.seq_end();
        let first = first.finish_fingerprint();

        let mut second = SchemaWriter::new();
        second.seq_begin("a");
        second.primitive("bc");
        second.seq_end();
        let second = second.finish_fingerprint();

        assert_ne!(first, second);
    }

    #[test]
    fn writer_token_boundaries_are_unambiguous() {
        let mut first = SchemaWriter::new();
        first.primitive("a");
        first.primitive("bc");
        let first = first.finish_fingerprint();

        let mut second = SchemaWriter::new();
        second.primitive("ab");
        second.primitive("c");
        let second = second.finish_fingerprint();

        assert_ne!(first, second);
    }

    #[test]
    fn map_schema_tokens_are_part_of_fingerprint() {
        let mut map = SchemaWriter::new();
        map.map_begin("Map");
        map.primitive("u32");
        map.primitive("u64");
        map.map_end();
        let map = map.finish_fingerprint();

        let mut seq = SchemaWriter::new();
        seq.seq_begin("Map");
        seq.primitive("u32");
        seq.primitive("u64");
        seq.seq_end();
        let seq = seq.finish_fingerprint();

        assert_ne!(map, seq);
    }

    #[test]
    fn default_writer_matches_new_writer() {
        let mut from_default = SchemaWriter::default();
        from_default.primitive("u32");

        let mut from_new = SchemaWriter::new();
        from_new.primitive("u32");

        assert_eq!(
            from_default.finish_fingerprint(),
            from_new.finish_fingerprint()
        );
    }

    #[test]
    fn debug_output_names_writer_without_exposing_hasher_state() {
        assert_eq!(format!("{:?}", SchemaWriter::new()), "SchemaWriter { .. }");
    }
}
