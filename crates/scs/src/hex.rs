use std::fmt::{self, Write as _};

use anyhow::{Result, anyhow};

pub(crate) struct HexPreview<'a> {
    bytes: &'a [u8],
    limit: usize,
}

impl<'a> HexPreview<'a> {
    fn new(bytes: &'a [u8], limit: usize) -> Self {
        Self { bytes, limit }
    }
}

pub(crate) trait HexPreviewExt {
    fn preview(&self, limit: usize) -> HexPreview<'_>;
}

impl HexPreviewExt for [u8] {
    fn preview(&self, limit: usize) -> HexPreview<'_> {
        HexPreview::new(self, limit)
    }
}

impl fmt::Display for HexPreview<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        const HEX: &[u8; 16] = b"0123456789abcdef";
        let shown = self.bytes.len().min(self.limit);
        for byte in &self.bytes[..shown] {
            formatter.write_char(char::from(HEX[usize::from(byte >> 4)]))?;
            formatter.write_char(char::from(HEX[usize::from(byte & 0x0f)]))?;
        }
        if shown < self.bytes.len() {
            formatter.write_str("...")?;
        }
        Ok(())
    }
}

pub(crate) trait HexParseExt {
    fn parse_hex_bytes(&self) -> Result<Vec<u8>>;
}

impl HexParseExt for str {
    fn parse_hex_bytes(&self) -> Result<Vec<u8>> {
        let input = self
            .strip_prefix("0x")
            .or_else(|| self.strip_prefix("0X"))
            .unwrap_or(self);
        HexInput::new(input).parse()
    }
}

struct HexInput<'a> {
    input: &'a str,
}

impl<'a> HexInput<'a> {
    fn new(input: &'a str) -> Self {
        Self { input }
    }

    fn parse(self) -> Result<Vec<u8>> {
        let mut bytes = Vec::with_capacity(self.input.len() / 2);
        let mut high_nibble = None;
        for byte in self.input.bytes() {
            if is_hex_separator(byte) {
                continue;
            }
            let nibble = hex_digit_value(byte)
                .ok_or_else(|| anyhow!("hex input contains an invalid digit"))?;
            if let Some(high) = high_nibble.take() {
                bytes.push((high << 4) | nibble);
            } else {
                high_nibble = Some(nibble);
            }
        }
        if high_nibble.is_some() {
            return Err(anyhow!("hex input has an odd length"));
        }
        Ok(bytes)
    }
}

fn hex_digit_value(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn is_hex_separator(byte: u8) -> bool {
    matches!(byte, b'_' | b':' | b'-' | b' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hex_accepts_prefix_and_common_separators() {
        assert_eq!(
            "0x0102:03-04_05"
                .parse_hex_bytes()
                .expect("hex should parse"),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn hex_preview_marks_truncation() {
        let bytes = [0xaa, 0xbb, 0xcc];
        assert_eq!(bytes.preview(2).to_string(), "aabb...");
    }
}
