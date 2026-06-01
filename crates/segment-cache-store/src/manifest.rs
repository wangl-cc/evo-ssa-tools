use std::{
    fs::{self, File},
    io::Write,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use crate::{
    error::{Error, Result},
    options::{StoreOptions, ValueLayout},
};

pub(crate) const MANIFEST_VERSION: u32 = 1;
pub(crate) const SHARD_ALGORITHM: &str = "lexicographic-prefix-v1";

const MANIFEST_MAGIC: &str = "segment-cache-store manifest v1";
const MANIFEST_FILE_NAME: &str = "MANIFEST";

#[derive(Clone, Debug)]
pub(crate) struct SegmentManifestEntry {
    pub file_name: String,
    pub min_key: Vec<u8>,
    pub max_key: Vec<u8>,
    pub record_count: u64,
    pub created_at_unix_millis: u128,
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
        for shard in &self.shards {
            let mut previous_max: Option<&[u8]> = None;
            for entry in shard {
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
                previous_max = Some(&entry.max_key);
            }
        }
        Ok(())
    }
}

pub(crate) fn manifest_path(root: &Path) -> PathBuf {
    root.join(MANIFEST_FILE_NAME)
}

pub(crate) fn load_manifest(root: &Path) -> Result<Option<StoreManifest>> {
    let path = manifest_path(root);
    if !path.exists() {
        return Ok(None);
    }
    parse_manifest(&fs::read_to_string(path)?).map(Some)
}

pub(crate) fn store_manifest(root: &Path, manifest: &StoreManifest) -> Result<()> {
    let path = manifest_path(root);
    let tmp_path = root.join("tmp").join(format!("{MANIFEST_FILE_NAME}.tmp"));
    let bytes = encode_manifest(manifest);
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(bytes.as_bytes())?;
        file.sync_all()?;
    }
    fs::rename(&tmp_path, &path)?;
    sync_dir(root)?;
    Ok(())
}

pub(crate) fn ensure_store_dirs(root: &Path, shard_count: usize) -> Result<()> {
    fs::create_dir_all(root.join("tmp"))?;
    let shards_root = root.join("shards");
    fs::create_dir_all(&shards_root)?;
    for shard in 0..shard_count {
        fs::create_dir_all(shards_root.join(shard.to_string()).join("segments"))?;
    }
    Ok(())
}

pub(crate) fn segment_dir(root: &Path, shard: usize) -> PathBuf {
    root.join("shards").join(shard.to_string()).join("segments")
}

pub(crate) fn temp_segment_path(root: &Path, shard: usize, segment_id: u64) -> PathBuf {
    root.join("tmp")
        .join(format!("segment-{shard:04}-{segment_id:020}.seg.tmp"))
}

pub(crate) fn final_segment_path(root: &Path, shard: usize, file_name: &str) -> PathBuf {
    segment_dir(root, shard).join(file_name)
}

pub(crate) fn next_segment_file_name(segment_id: u64) -> String {
    format!("segment-{segment_id:020}.seg")
}

pub(crate) fn now_unix_millis() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time should be after epoch")
        .as_millis()
}

pub(crate) fn sync_dir(path: &Path) -> Result<()> {
    let dir = File::open(path)?;
    dir.sync_all()?;
    Ok(())
}

fn encode_manifest(manifest: &StoreManifest) -> String {
    let mut out = String::new();
    push_line(&mut out, MANIFEST_MAGIC);
    push_line(&mut out, &format!("version={}", manifest.version));
    push_line(&mut out, &format!("key_len={}", manifest.key_len));
    push_line(
        &mut out,
        &format!(
            "value_layout={}",
            encode_value_layout(manifest.value_layout)
        ),
    );
    push_line(&mut out, &format!("shard_count={}", manifest.shard_count));
    push_line(
        &mut out,
        &format!("shard_key_offset={}", manifest.shard_key_offset),
    );
    push_line(
        &mut out,
        &format!("target_block_size={}", manifest.target_block_size),
    );
    push_line(
        &mut out,
        &format!("shard_algorithm={}", manifest.shard_algorithm),
    );
    push_line(
        &mut out,
        &format!("next_segment_id={}", manifest.next_segment_id),
    );
    for (shard_id, shard) in manifest.shards.iter().enumerate() {
        push_line(&mut out, &format!("[shard {shard_id}]"));
        for entry in shard {
            push_line(
                &mut out,
                &format!(
                    "segment\t{}\t{}\t{}\t{}\t{}",
                    entry.file_name,
                    encode_hex(&entry.min_key),
                    encode_hex(&entry.max_key),
                    entry.record_count,
                    entry.created_at_unix_millis,
                ),
            );
        }
    }
    out
}

fn parse_manifest(input: &str) -> Result<StoreManifest> {
    let mut lines = input.lines();
    let Some(magic) = lines.next() else {
        return manifest_parse("empty manifest");
    };
    if magic != MANIFEST_MAGIC {
        return manifest_parse("unsupported manifest magic");
    }

    let version = parse_required_value::<u32>(&mut lines, "version")?;
    let key_len = parse_required_value::<usize>(&mut lines, "key_len")?;
    let value_layout = parse_value_layout(parse_required_str(&mut lines, "value_layout")?)?;
    let shard_count = parse_required_value::<usize>(&mut lines, "shard_count")?;
    let shard_key_offset = parse_required_value::<usize>(&mut lines, "shard_key_offset")?;
    let target_block_size = parse_required_value::<usize>(&mut lines, "target_block_size")?;
    let shard_algorithm = parse_required_str(&mut lines, "shard_algorithm")?.to_owned();
    let next_segment_id = parse_required_value::<u64>(&mut lines, "next_segment_id")?;
    let mut shards = vec![Vec::new(); shard_count];
    let mut current_shard = None;

    for line in lines {
        if line.is_empty() {
            continue;
        }
        if let Some(shard_id) = parse_shard_header(line)? {
            if shard_id >= shard_count {
                return manifest_parse("shard id out of range");
            }
            current_shard = Some(shard_id);
            continue;
        }
        let Some(shard_id) = current_shard else {
            return manifest_parse("segment entry before shard header");
        };
        shards[shard_id].push(parse_segment_entry(line)?);
    }

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

fn push_line(out: &mut String, line: &str) {
    out.push_str(line);
    out.push('\n');
}

fn parse_required_str<'a>(lines: &mut std::str::Lines<'a>, key: &'static str) -> Result<&'a str> {
    let Some(line) = lines.next() else {
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

fn parse_required_value<T>(lines: &mut std::str::Lines<'_>, key: &'static str) -> Result<T>
where
    T: std::str::FromStr,
{
    parse_required_str(lines, key)?
        .parse()
        .map_err(|_| Error::ManifestParse {
            reason: format!("invalid value for {key}"),
        })
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

fn parse_segment_entry(line: &str) -> Result<SegmentManifestEntry> {
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
    if fields.next().is_some() {
        return manifest_parse("too many segment fields");
    }
    Ok(SegmentManifestEntry {
        file_name: file_name.to_owned(),
        min_key: decode_hex(min_key)?,
        max_key: decode_hex(max_key)?,
        record_count: record_count.parse().map_err(|_| Error::ManifestParse {
            reason: "invalid segment record count".to_owned(),
        })?,
        created_at_unix_millis: created_at_unix_millis.parse().map_err(|_| {
            Error::ManifestParse {
                reason: "invalid segment creation time".to_owned(),
            }
        })?,
    })
}

fn encode_value_layout(value_layout: ValueLayout) -> String {
    match value_layout {
        ValueLayout::Variable => "variable".to_owned(),
        ValueLayout::Fixed { value_len } => format!("fixed:{value_len}"),
    }
}

fn parse_value_layout(value: &str) -> Result<ValueLayout> {
    if value == "variable" {
        return Ok(ValueLayout::Variable);
    }
    if let Some(value_len) = value.strip_prefix("fixed:") {
        let value_len = value_len.parse().map_err(|_| Error::ManifestParse {
            reason: "invalid fixed value length".to_owned(),
        })?;
        return Ok(ValueLayout::Fixed { value_len });
    }
    manifest_parse("invalid value layout")
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(char::from(HEX[usize::from(byte >> 4)]));
        out.push(char::from(HEX[usize::from(byte & 0x0f)]));
    }
    out
}

fn decode_hex(value: &str) -> Result<Vec<u8>> {
    if !value.len().is_multiple_of(2) {
        return manifest_parse("hex value has odd length");
    }
    let mut bytes = Vec::with_capacity(value.len() / 2);
    for pair in value.as_bytes().chunks_exact(2) {
        bytes.push((decode_hex_nibble(pair[0])? << 4) | decode_hex_nibble(pair[1])?);
    }
    Ok(bytes)
}

fn decode_hex_nibble(value: u8) -> Result<u8> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => manifest_parse("invalid hex digit"),
    }
}

fn manifest_parse<T>(reason: impl Into<String>) -> Result<T> {
    Err(Error::ManifestParse {
        reason: reason.into(),
    })
}
