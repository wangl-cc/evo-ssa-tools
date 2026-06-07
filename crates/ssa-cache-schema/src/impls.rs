mod collections;
mod primitives;
mod sequences;
mod text;
mod transparent;

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use crate::schema_fingerprint;

    #[test]
    fn fingerprint_is_deterministic() {
        assert_eq!(
            schema_fingerprint::<(u32, bool)>(),
            schema_fingerprint::<(u32, bool)>()
        );
    }
}
