//! Store manifest and filesystem layout.
//!
//! The manifest is the only source of truth for visible segment files. It is a
//! deliberately small, line-oriented v1 format rather than a generic
//! serialization format. Keeping parsing and encoding on `StoreManifest` and
//! `SegmentManifestEntry` makes the review contract explicit: manifest metadata
//! owns manifest I/O; segment file code only consumes the resulting entries.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{Read, Write},
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crc32c::crc32c_append;

use crate::{
    error::{Error, Result},
    options::{StoreOptions, ValueLayout},
};

pub(crate) const MANIFEST_VERSION: u32 = 1;
pub(crate) const SHARD_ALGORITHM: &str = "lexicographic-prefix-v1";

const MANIFEST_MAGIC: &str = "segment-cache-store manifest v1";
const MANIFEST_FILE_NAME: &str = "MANIFEST";

/// Path helper for all files owned by one store root.
///
/// Keeping path construction here prevents the storage logic from hand-rolling
/// directory names and makes crash-recovery cases easier to audit.
pub(crate) struct StorePaths<'a> {
    root: &'a Path,
}

impl<'a> StorePaths<'a> {
    pub(crate) fn new(root: &'a Path) -> Self {
        Self { root }
    }

    pub(crate) fn manifest(&self) -> PathBuf {
        self.root.join(MANIFEST_FILE_NAME)
    }

    pub(crate) fn temp_manifest(&self) -> PathBuf {
        self.root
            .join("tmp")
            .join(format!("{MANIFEST_FILE_NAME}.tmp"))
    }

    pub(crate) fn ensure_dirs(&self, shard_count: usize) -> Result<()> {
        fs::create_dir_all(self.root.join("tmp"))?;
        let shards_root = self.root.join("shards");
        fs::create_dir_all(&shards_root)?;
        for shard in 0..shard_count {
            fs::create_dir_all(shards_root.join(shard.to_string()).join("segments"))?;
        }
        Ok(())
    }

    pub(crate) fn segment_dir(&self, shard: usize) -> PathBuf {
        self.root
            .join("shards")
            .join(shard.to_string())
            .join("segments")
    }

    pub(crate) fn temp_segment(&self, shard: usize, segment_id: u64) -> PathBuf {
        self.root
            .join("tmp")
            .join(format!("segment-{shard:04}-{segment_id:020}.seg.tmp"))
    }

    pub(crate) fn final_segment(&self, shard: usize, file_name: &str) -> PathBuf {
        self.segment_dir(shard).join(file_name)
    }

    pub(crate) fn sync_root(&self) -> Result<()> {
        sync_dir(self.root)
    }

    pub(crate) fn sync_segment_dir(&self, shard: usize) -> Result<()> {
        sync_dir(&self.segment_dir(shard))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct SegmentManifestEntry {
    pub file_name: String,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub record_count: u64,
    pub created_at_unix_millis: u128,
    pub file_fingerprint: SegmentFileFingerprint,
}

impl SegmentManifestEntry {
    fn encode_line(&self) -> String {
        format!(
            "segment\t{}\t{}\t{}\t{}\t{}\t{}\t{:08x}",
            self.file_name,
            HexBytes::encode(&self.min_key),
            HexBytes::encode(&self.max_key),
            self.record_count,
            self.created_at_unix_millis,
            self.file_fingerprint.len,
            self.file_fingerprint.crc32c,
        )
    }

    fn parse_line(line: &str) -> Result<Self> {
        let mut fields = line.split('\t');
        if fields.next() != Some("segment") {
            return manifest_parse("malformed segment entry");
        }
        let Some(file_name) = fields.next() else {
            return manifest_parse("missing segment file name");
        };
        let Some(min_key) = fields.next() else {
            return manifest_parse("missing segment min key");
        };
        let Some(max_key) = fields.next() else {
            return manifest_parse("missing segment max key");
        };
        let Some(record_count) = fields.next() else {
            return manifest_parse("missing segment record count");
        };
        let Some(created_at_unix_millis) = fields.next() else {
            return manifest_parse("missing segment creation time");
        };
        let Some(file_len) = fields.next() else {
            return manifest_parse("missing segment file length");
        };
        let Some(file_crc32c) = fields.next() else {
            return manifest_parse("missing segment file checksum");
        };
        if fields.next().is_some() {
            return manifest_parse("too many segment fields");
        }
        Ok(Self {
            file_name: file_name.to_owned(),
            min_key: HexBytes::decode(min_key)?,
            max_key: HexBytes::decode(max_key)?,
            record_count: record_count.parse().map_err(|_| Error::ManifestParse {
                reason: "invalid segment record count".to_owned(),
            })?,
            created_at_unix_millis: created_at_unix_millis.parse().map_err(|_| {
                Error::ManifestParse {
                    reason: "invalid segment creation time".to_owned(),
                }
            })?,
            file_fingerprint: SegmentFileFingerprint {
                len: file_len.parse().map_err(|_| Error::ManifestParse {
                    reason: "invalid segment file length".to_owned(),
                })?,
                crc32c: u32::from_str_radix(file_crc32c, 16).map_err(|_| Error::ManifestParse {
                    reason: "invalid segment file checksum".to_owned(),
                })?,
            },
        })
    }

    pub(crate) fn matches_segment_footer(
        &self,
        min_key: &[u8],
        max_key: &[u8],
        record_count: u64,
    ) -> bool {
        self.min_key == min_key && self.max_key == max_key && self.record_count == record_count
    }
}

/// Stable identity metadata for an immutable segment file.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SegmentFileFingerprint {
    pub len: u64,
    pub crc32c: u32,
}

impl SegmentFileFingerprint {
    pub(crate) fn read_len_from_path(path: &Path) -> Result<Option<u64>> {
        match fs::metadata(path) {
            Ok(metadata) => Ok(Some(metadata.len())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    pub(crate) fn read_from_path(path: &Path) -> Result<Option<Self>> {
        let mut file = match File::open(path) {
            Ok(file) => file,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(error) => return Err(error.into()),
        };
        let len = file.metadata()?.len();
        let mut crc32c = 0;
        let mut buffer = [0u8; 64 * 1024];
        loop {
            let read = file.read(&mut buffer)?;
            if read == 0 {
                break;
            }
            crc32c = crc32c_append(crc32c, &buffer[..read]);
        }
        Ok(Some(Self { len, crc32c }))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StoreManifest {
    pub version: u32,
    pub key_len: usize,
    pub value_layout: ValueLayout,
    pub shard_count: usize,
    pub shard_key_offset: usize,
    pub target_block_size: usize,
    pub shard_algorithm: String,
    pub next_segment_id: u64,
    pub shards: Vec<Vec<SegmentManifestEntry>>,
}

impl StoreManifest {
    pub(crate) fn new(options: &StoreOptions) -> Self {
        Self {
            version: MANIFEST_VERSION,
            key_len: options.key_len,
            value_layout: options.value_layout,
            shard_count: options.shard_count,
            shard_key_offset: options.shard_key_offset,
            target_block_size: options.target_block_size,
            shard_algorithm: SHARD_ALGORITHM.to_owned(),
            next_segment_id: 0,
            shards: vec![Vec::new(); options.shard_count],
        }
    }

    pub(crate) fn load(root: &Path) -> Result<Option<Self>> {
        let paths = StorePaths::new(root);
        let path = paths.manifest();
        if !path.exists() {
            return Ok(None);
        }
        Self::parse(&fs::read_to_string(path)?).map(Some)
    }

    pub(crate) fn store(&self, root: &Path) -> Result<()> {
        let paths = StorePaths::new(root);
        let tmp_path = paths.temp_manifest();
        {
            let mut file = File::create(&tmp_path)?;
            file.write_all(self.encode().as_bytes())?;
            file.sync_all()?;
        }
        fs::rename(tmp_path, paths.manifest())?;
        paths.sync_root()?;
        Ok(())
    }

    pub(crate) fn encode(&self) -> String {
        let mut out = String::new();
        out.push_line(MANIFEST_MAGIC);
        out.push_line(&format!("version={}", self.version));
        out.push_line(&format!("key_len={}", self.key_len));
        out.push_line(&format!(
            "value_layout={}",
            self.value_layout.encode_manifest_value()
        ));
        out.push_line(&format!("shard_count={}", self.shard_count));
        out.push_line(&format!("shard_key_offset={}", self.shard_key_offset));
        out.push_line(&format!("target_block_size={}", self.target_block_size));
        out.push_line(&format!("shard_algorithm={}", self.shard_algorithm));
        out.push_line(&format!("next_segment_id={}", self.next_segment_id));
        for (shard_id, shard) in self.shards.iter().enumerate() {
            out.push_line(&format!("[shard {shard_id}]"));
            for entry in shard {
                out.push_line(&entry.encode_line());
            }
        }
        out
    }

    pub(crate) fn parse(input: &str) -> Result<Self> {
        ManifestParser::new(input).parse()
    }

    pub(crate) fn validate_options(&self, options: &StoreOptions) -> Result<()> {
        if self.version != MANIFEST_VERSION {
            return Err(Error::UnsupportedFormatVersion {
                version: self.version,
            });
        }
        if self.key_len != options.key_len {
            return Err(Error::ManifestMismatch { reason: "key_len" });
        }
        if self.value_layout != options.value_layout {
            return Err(Error::ManifestMismatch {
                reason: "value_layout",
            });
        }
        if self.shard_count != options.shard_count {
            return Err(Error::ManifestMismatch {
                reason: "shard_count",
            });
        }
        if self.shard_key_offset != options.shard_key_offset {
            return Err(Error::ManifestMismatch {
                reason: "shard_key_offset",
            });
        }
        if self.shard_algorithm != SHARD_ALGORITHM {
            return Err(Error::ManifestMismatch {
                reason: "shard_algorithm",
            });
        }
        if self.shards.len() != self.shard_count {
            return Err(Error::ManifestMismatch {
                reason: "shards_len",
            });
        }
        let mut seen_files = BTreeSet::new();
        let mut max_segment_id: Option<u64> = None;
        for shard in &self.shards {
            let mut previous_max: Option<&[u8]> = None;
            for entry in shard {
                self.validate_segment_entry(entry, previous_max)?;
                let Some(segment_id) = segment_id_from_file_name(&entry.file_name) else {
                    return Err(Error::ManifestMismatch {
                        reason: "segment_file_name",
                    });
                };
                if !seen_files.insert(entry.file_name.as_str()) {
                    return Err(Error::ManifestMismatch {
                        reason: "duplicate_segment_file",
                    });
                }
                max_segment_id = Some(max_segment_id.map_or(segment_id, |max| max.max(segment_id)));
                previous_max = Some(&entry.max_key);
            }
        }
        if let Some(max_segment_id) = max_segment_id
            && self.next_segment_id <= max_segment_id
        {
            return Err(Error::ManifestMismatch {
                reason: "next_segment_id",
            });
        }
        Ok(())
    }

    fn validate_segment_entry(
        &self,
        entry: &SegmentManifestEntry,
        previous_max: Option<&[u8]>,
    ) -> Result<()> {
        if entry.min_key.len() != self.key_len || entry.max_key.len() != self.key_len {
            return Err(Error::ManifestMismatch {
                reason: "segment_key_len",
            });
        }
        if entry.min_key > entry.max_key {
            return Err(Error::ManifestMismatch {
                reason: "segment_key_range",
            });
        }
        if let Some(previous_max) = previous_max
            && entry.min_key.as_slice() <= previous_max
        {
            return Err(Error::ManifestMismatch {
                reason: "segment_overlap",
            });
        }
        Ok(())
    }
}

struct ManifestParser<'a> {
    lines: std::str::Lines<'a>,
}

impl<'a> ManifestParser<'a> {
    fn new(input: &'a str) -> Self {
        Self {
            lines: input.lines(),
        }
    }

    fn parse(mut self) -> Result<StoreManifest> {
        self.expect_magic()?;
        let version = self.required_value::<u32>("version")?;
        let key_len = self.required_value::<usize>("key_len")?;
        let value_layout = ValueLayout::parse_manifest_value(self.required_str("value_layout")?)?;
        let shard_count = self.required_value::<usize>("shard_count")?;
        let shard_key_offset = self.required_value::<usize>("shard_key_offset")?;
        let target_block_size = self.required_value::<usize>("target_block_size")?;
        let shard_algorithm = self.required_str("shard_algorithm")?.to_owned();
        let next_segment_id = self.required_value::<u64>("next_segment_id")?;
        let shards = self.parse_shards(shard_count)?;

        Ok(StoreManifest {
            version,
            key_len,
            value_layout,
            shard_count,
            shard_key_offset,
            target_block_size,
            shard_algorithm,
            next_segment_id,
            shards,
        })
    }

    fn expect_magic(&mut self) -> Result<()> {
        let Some(magic) = self.lines.next() else {
            return manifest_parse("empty manifest");
        };
        if magic != MANIFEST_MAGIC {
            return manifest_parse("unsupported manifest magic");
        }
        Ok(())
    }

    fn required_str(&mut self, key: &'static str) -> Result<&'a str> {
        let Some(line) = self.lines.next() else {
            return manifest_parse("missing required manifest field");
        };
        let Some((actual_key, value)) = line.split_once('=') else {
            return manifest_parse("malformed manifest field");
        };
        if actual_key != key {
            return manifest_parse("unexpected manifest field order");
        }
        Ok(value)
    }

    fn required_value<T>(&mut self, key: &'static str) -> Result<T>
    where
        T: std::str::FromStr,
    {
        self.required_str(key)?
            .parse()
            .map_err(|_| Error::ManifestParse {
                reason: format!("invalid value for {key}"),
            })
    }

    fn parse_shards(&mut self, shard_count: usize) -> Result<Vec<Vec<SegmentManifestEntry>>> {
        let mut shards = vec![Vec::new(); shard_count];
        let mut current_shard = None;

        for line in self.lines.by_ref() {
            if line.is_empty() {
                continue;
            }
            if let Some(shard_id) = Self::parse_shard_header(line)? {
                if shard_id >= shard_count {
                    return manifest_parse("shard id out of range");
                }
                current_shard = Some(shard_id);
                continue;
            }
            let Some(shard_id) = current_shard else {
                return manifest_parse("segment entry before shard header");
            };
            shards[shard_id].push(SegmentManifestEntry::parse_line(line)?);
        }

        Ok(shards)
    }

    fn parse_shard_header(line: &str) -> Result<Option<usize>> {
        let Some(inner) = line.strip_prefix("[shard ") else {
            return Ok(None);
        };
        let Some(id) = inner.strip_suffix(']') else {
            return manifest_parse("malformed shard header");
        };
        id.parse().map(Some).map_err(|_| Error::ManifestParse {
            reason: "invalid shard id".to_owned(),
        })
    }
}

trait ManifestStringExt {
    fn push_line(&mut self, line: &str);
}

impl ManifestStringExt for String {
    fn push_line(&mut self, line: &str) {
        self.push_str(line);
        self.push('\n');
    }
}

struct HexBytes;

impl HexBytes {
    fn encode(bytes: &[u8]) -> String {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let mut out = String::with_capacity(bytes.len() * 2);
        for byte in bytes {
            out.push(char::from(HEX[usize::from(byte >> 4)]));
            out.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
        out
    }

    fn decode(value: &str) -> Result<Vec<u8>> {
        if !value.len().is_multiple_of(2) {
            return manifest_parse("hex value has odd length");
        }
        let mut bytes = Vec::with_capacity(value.len() / 2);
        for pair in value.as_bytes().chunks_exact(2) {
            bytes.push((Self::decode_nibble(pair[0])? << 4) | Self::decode_nibble(pair[1])?);
        }
        Ok(bytes)
    }

    fn decode_nibble(value: u8) -> Result<u8> {
        match value {
            b'0'..=b'9' => Ok(value - b'0'),
            b'a'..=b'f' => Ok(value - b'a' + 10),
            b'A'..=b'F' => Ok(value - b'A' + 10),
            _ => manifest_parse("invalid hex digit"),
        }
    }
}

pub(crate) fn next_segment_file_name(segment_id: u64) -> String {
    format!("segment-{segment_id:020}.seg")
}

fn segment_id_from_file_name(file_name: &str) -> Option<u64> {
    file_name
        .strip_prefix("segment-")?
        .strip_suffix(".seg")?
        .parse()
        .ok()
}

pub(crate) fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_millis()
}

fn sync_dir(path: &Path) -> Result<()> {
    let dir = File::open(path)?;
    dir.sync_all()?;
    Ok(())
}

fn manifest_parse<T>(reason: impl Into<String>) -> Result<T> {
    Err(Error::ManifestParse {
        reason: reason.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::StoreManifest;
    use crate::{Error, StoreOptions};

    fn manifest_with_segment_line(segment_line: &str) -> String {
        format!(
            "\
segment-cache-store manifest v1
version=1
key_len=16
value_layout=variable
shard_count=1
shard_key_offset=16
target_block_size=256
shard_algorithm=lexicographic-prefix-v1
next_segment_id=0
[shard 0]
{segment_line}
"
        )
    }

    mod parser {
        use super::*;

        #[test]
        fn rejects_malformed_segment_entries() {
            for manifest in [
                manifest_with_segment_line("not-segment"),
                manifest_with_segment_line("segment\tfile"),
                manifest_with_segment_line(
                    "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t\
                     00000000000000000000000000000001\tnot-a-count\t0\t1\t00000000",
                ),
                manifest_with_segment_line(
                    "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t\
                     00000000000000000000000000000001\t1\tnot-a-time\t1\t00000000",
                ),
                manifest_with_segment_line(
                    "segment\tsegment-00000000000000000000.seg\txyz\t00000000000000000000000000000001\t1\t0\t1\t00000000",
                ),
                manifest_with_segment_line(
                    "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t\
                     00000000000000000000000000000001\t1\t0\t1\tnot-crc",
                ),
                manifest_with_segment_line(
                    "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t\
                     00000000000000000000000000000001\t1\t0\tnot-len\t00000000",
                ),
                manifest_with_segment_line(
                    "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t\
                     00000000000000000000000000000001\t1\t0\t1\t00000000\textra",
                ),
            ] {
                assert!(matches!(
                    StoreManifest::parse(&manifest),
                    Err(Error::ManifestParse { .. })
                ));
            }
        }

        #[test]
        fn rejects_invalid_structure() {
            assert!(matches!(
                StoreManifest::parse(""),
                Err(Error::ManifestParse { .. })
            ));
            assert!(matches!(
                StoreManifest::parse(
                    "\
segment-cache-store manifest v1
version=1
key_len=16
value_layout=variable
shard_count=1
shard_key_offset=16
target_block_size=256
shard_algorithm=lexicographic-prefix-v1
next_segment_id=0
	segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t00000000000000000000000000000001\t1\t0\t1\t00000000
"
                ),
                Err(Error::ManifestParse { .. })
            ));
        }
    }

    mod validation {
        use super::*;

        #[test]
        fn rejects_invalid_segment_ranges() {
            let reversed = StoreManifest::parse(&manifest_with_segment_line(
                "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000002\t\
             00000000000000000000000000000001\t1\t0\t1\t00000000",
            ))
            .expect("manifest syntax should parse");
            let options = StoreOptions::new("", 16)
                .with_shard_count(1)
                .with_shard_key_offset(16);
            assert!(matches!(
                reversed.validate_options(&options),
                Err(Error::ManifestMismatch {
                    reason: "segment_key_range"
                })
            ));

            let mut overlapping = StoreManifest::parse(&manifest_with_segment_line(
                "segment\tsegment-00000000000000000000.seg\t00000000000000000000000000000000\t\
             00000000000000000000000000000001\t1\t0\t1\t00000000",
            ))
            .expect("manifest syntax should parse");
            let duplicate = overlapping.shards[0][0].clone();
            overlapping.shards[0].push(duplicate);
            assert!(matches!(
                overlapping.validate_options(&options),
                Err(Error::ManifestMismatch {
                    reason: "segment_overlap"
                })
            ));
        }
    }
}
