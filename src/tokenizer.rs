use std::collections::HashMap;
use std::fmt;
use std::fs;
use std::path::Path;

use fancy_regex::Regex;
use serde::Deserialize;

const BYTE_PATTERN: &str =
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

#[derive(Debug)]
pub enum TokenizerError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Regex(Box<fancy_regex::Error>),
    Unsupported(&'static str),
    InvalidVocabulary(String),
    InvalidToken(u32),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "could not read tokenizer: {error}"),
            Self::Json(error) => write!(f, "invalid tokenizer JSON: {error}"),
            Self::Regex(error) => write!(f, "tokenizer pattern failed: {error}"),
            Self::Unsupported(option) => write!(f, "unsupported tokenizer option: {option}"),
            Self::InvalidVocabulary(piece) => write!(f, "tokenizer has no token for {piece:?}"),
            Self::InvalidToken(id) => write!(f, "tokenizer has no token ID {id}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

impl From<std::io::Error> for TokenizerError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<serde_json::Error> for TokenizerError {
    fn from(error: serde_json::Error) -> Self {
        Self::Json(error)
    }
}

impl From<fancy_regex::Error> for TokenizerError {
    fn from(error: fancy_regex::Error) -> Self {
        Self::Regex(Box::new(error))
    }
}

#[derive(Deserialize)]
struct TokenizerSpec {
    normalizer: Option<serde_json::Value>,
    post_processor: Option<serde_json::Value>,
    truncation: Option<serde_json::Value>,
    padding: Option<serde_json::Value>,
    pre_tokenizer: PreTokenizerSpec,
    decoder: DecoderSpec,
    model: ModelSpec,
    added_tokens: Vec<AddedToken>,
}

#[derive(Deserialize)]
struct PreTokenizerSpec {
    #[serde(rename = "type")]
    kind: String,
    pretokenizers: Vec<serde_json::Value>,
}

#[derive(Deserialize)]
struct DecoderSpec {
    #[serde(rename = "type")]
    kind: String,
}

#[derive(Deserialize)]
struct ModelSpec {
    #[serde(rename = "type")]
    kind: String,
    dropout: Option<f32>,
    unk_token: Option<String>,
    continuing_subword_prefix: Option<String>,
    end_of_word_suffix: Option<String>,
    fuse_unk: bool,
    byte_fallback: bool,
    ignore_merges: bool,
    vocab: HashMap<String, u32>,
    merges: Vec<String>,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
    special: bool,
    single_word: bool,
    lstrip: bool,
    rstrip: bool,
    normalized: bool,
}

/// Byte-level BPE for the supported SmolLM2 tokenizer layout.
pub struct ByteBpeTokenizer {
    vocab: HashMap<String, u32>,
    tokens: Vec<Option<String>>,
    merge_rank: HashMap<(String, String), usize>,
    special: Vec<(String, u32)>,
    byte_to_char: [char; 256],
    char_to_byte: HashMap<char, u8>,
    pattern: Regex,
}

/// Decode tokens as UTF-8 text without emitting incomplete characters.
pub struct ByteBpeDecoder {
    pending: Vec<u8>,
}

impl Default for ByteBpeDecoder {
    fn default() -> Self {
        Self::new()
    }
}

impl ByteBpeDecoder {
    pub fn new() -> Self {
        Self {
            pending: Vec::new(),
        }
    }

    pub fn push(
        &mut self,
        tokenizer: &ByteBpeTokenizer,
        id: u32,
    ) -> Result<String, TokenizerError> {
        self.pending.extend(tokenizer.decode_bytes(&[id], true)?);
        let mut output = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(text) => {
                    output.push_str(text);
                    self.pending.clear();
                    break;
                }
                Err(error) => {
                    let valid = error.valid_up_to();
                    output.push_str(
                        std::str::from_utf8(&self.pending[..valid])
                            .expect("UTF-8 parser identified a valid prefix"),
                    );
                    self.pending.drain(..valid);
                    match error.error_len() {
                        Some(length) => {
                            output.push('\u{fffd}');
                            self.pending.drain(..length);
                        }
                        None => break,
                    }
                }
            }
        }
        Ok(output)
    }

    pub fn finish(self) -> String {
        String::from_utf8_lossy(&self.pending).into_owned()
    }
}

impl ByteBpeTokenizer {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, TokenizerError> {
        let spec: TokenizerSpec = serde_json::from_slice(&fs::read(path)?)?;
        if spec.normalizer.is_some()
            || spec.post_processor.is_some()
            || spec.truncation.is_some()
            || spec.padding.is_some()
        {
            return Err(TokenizerError::Unsupported(
                "normalization or post-processing",
            ));
        }
        if spec.pre_tokenizer.kind != "Sequence" || spec.pre_tokenizer.pretokenizers.len() != 2 {
            return Err(TokenizerError::Unsupported("pre-tokenizer sequence"));
        }
        let digits = &spec.pre_tokenizer.pretokenizers[0];
        let byte_level = &spec.pre_tokenizer.pretokenizers[1];
        if digits["type"] != "Digits"
            || digits["individual_digits"] != true
            || byte_level["type"] != "ByteLevel"
            || byte_level["add_prefix_space"] != false
            || byte_level["use_regex"] != true
            || spec.decoder.kind != "ByteLevel"
        {
            return Err(TokenizerError::Unsupported("pre-tokenizer or decoder"));
        }
        let model = spec.model;
        if model.kind != "BPE"
            || model.dropout.is_some()
            || model.unk_token.is_some()
            || model.continuing_subword_prefix.is_some()
            || model.end_of_word_suffix.is_some()
            || model.fuse_unk
            || model.byte_fallback
            || model.ignore_merges
        {
            return Err(TokenizerError::Unsupported("BPE model options"));
        }
        let mut tokens = vec![None; model.vocab.len()];
        for (piece, &id) in &model.vocab {
            let slot = tokens
                .get_mut(id as usize)
                .ok_or_else(|| TokenizerError::InvalidVocabulary(piece.clone()))?;
            if slot.replace(piece.clone()).is_some() {
                return Err(TokenizerError::InvalidVocabulary(piece.clone()));
            }
        }
        let mut special = Vec::with_capacity(spec.added_tokens.len());
        for token in spec.added_tokens {
            if !token.special
                || token.single_word
                || token.lstrip
                || token.rstrip
                || token.normalized
            {
                return Err(TokenizerError::Unsupported("added token options"));
            }
            if tokens.get(token.id as usize).and_then(Option::as_deref)
                != Some(token.content.as_str())
            {
                return Err(TokenizerError::InvalidVocabulary(token.content));
            }
            special.push((token.content, token.id));
        }
        Self::build(model.vocab, tokens, model.merges, special)
    }

    pub(crate) fn from_gguf_parts(
        pieces: Vec<String>,
        merges: Vec<String>,
        token_types: Vec<i32>,
    ) -> Result<Self, TokenizerError> {
        if pieces.len() != token_types.len() || pieces.len() > u32::MAX as usize {
            return Err(TokenizerError::Unsupported("GGUF token types"));
        }
        let mut vocab = HashMap::with_capacity(pieces.len());
        let mut tokens = Vec::with_capacity(pieces.len());
        let mut special = Vec::new();
        for (index, (piece, kind)) in pieces.into_iter().zip(token_types).enumerate() {
            match kind {
                1 => {}
                3 => special.push((piece.clone(), index as u32)),
                _ => return Err(TokenizerError::Unsupported("GGUF token type")),
            }
            if vocab.insert(piece.clone(), index as u32).is_some() {
                return Err(TokenizerError::InvalidVocabulary(piece));
            }
            tokens.push(Some(piece));
        }
        Self::build(vocab, tokens, merges, special)
    }

    fn build(
        vocab: HashMap<String, u32>,
        tokens: Vec<Option<String>>,
        merges: Vec<String>,
        mut special: Vec<(String, u32)>,
    ) -> Result<Self, TokenizerError> {
        let mut merge_rank = HashMap::with_capacity(merges.len());
        for (rank, merge) in merges.into_iter().enumerate() {
            let (left, right) = merge
                .split_once(' ')
                .ok_or_else(|| TokenizerError::InvalidVocabulary(merge.clone()))?;
            if merge_rank
                .insert((left.to_owned(), right.to_owned()), rank)
                .is_some()
            {
                return Err(TokenizerError::InvalidVocabulary(merge));
            }
        }
        special.sort_by_key(|token| std::cmp::Reverse(token.0.len()));
        let byte_to_char = byte_alphabet();
        let char_to_byte = byte_to_char
            .iter()
            .enumerate()
            .map(|(index, &letter)| (letter, index as u8))
            .collect();
        Ok(Self {
            vocab,
            tokens,
            merge_rank,
            special,
            byte_to_char,
            char_to_byte,
            pattern: Regex::new(BYTE_PATTERN)?,
        })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut result = Vec::new();
        let mut position = 0;
        while position < text.len() {
            let next = self
                .special
                .iter()
                .filter_map(|(content, id)| {
                    text[position..].find(content).map(|at| (at, content, id))
                })
                .min_by_key(|(at, content, _)| (*at, usize::MAX - content.len()));
            match next {
                Some((at, content, id)) => {
                    self.encode_plain(&text[position..position + at], &mut result)?;
                    result.push(*id);
                    position += at + content.len();
                }
                None => {
                    self.encode_plain(&text[position..], &mut result)?;
                    break;
                }
            }
        }
        Ok(result)
    }

    /// Encode user text without interpreting special-token spellings as control tokens.
    pub fn encode_plain_text(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let mut result = Vec::new();
        self.encode_plain(text, &mut result)?;
        Ok(result)
    }

    pub fn special_token_id(&self, content: &str) -> Option<u32> {
        self.special
            .iter()
            .find(|(token, _)| token == content)
            .map(|(_, id)| *id)
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> Result<String, TokenizerError> {
        Ok(String::from_utf8_lossy(&self.decode_bytes(ids, skip_special)?).into_owned())
    }

    pub fn decode_bytes(&self, ids: &[u32], skip_special: bool) -> Result<Vec<u8>, TokenizerError> {
        let mut bytes = Vec::new();
        for &id in ids {
            let piece = self
                .tokens
                .get(id as usize)
                .and_then(Option::as_deref)
                .ok_or(TokenizerError::InvalidToken(id))?;
            if self.special.iter().any(|(_, special_id)| *special_id == id) {
                if !skip_special {
                    bytes.extend_from_slice(piece.as_bytes());
                }
                continue;
            }
            for letter in piece.chars() {
                let byte = self
                    .char_to_byte
                    .get(&letter)
                    .copied()
                    .ok_or_else(|| TokenizerError::InvalidVocabulary(piece.to_owned()))?;
                bytes.push(byte);
            }
        }
        Ok(bytes)
    }

    fn encode_plain(&self, text: &str, result: &mut Vec<u32>) -> Result<(), TokenizerError> {
        let mut start = 0;
        for (at, character) in text.char_indices() {
            if character.is_numeric() {
                self.encode_byte_level(&text[start..at], result)?;
                self.encode_byte_level(&text[at..at + character.len_utf8()], result)?;
                start = at + character.len_utf8();
            }
        }
        self.encode_byte_level(&text[start..], result)
    }

    fn encode_byte_level(&self, text: &str, result: &mut Vec<u32>) -> Result<(), TokenizerError> {
        for matched in self.pattern.find_iter(text) {
            let part = matched?.as_str();
            let mut pieces: Vec<String> = part
                .as_bytes()
                .iter()
                .map(|&byte| self.byte_to_char[byte as usize].to_string())
                .collect();
            while pieces.len() > 1 {
                let best = pieces
                    .windows(2)
                    .enumerate()
                    .filter_map(|(at, pair)| {
                        self.merge_rank
                            .get(&(pair[0].clone(), pair[1].clone()))
                            .map(|&rank| (rank, at))
                    })
                    .min();
                let Some((_, at)) = best else { break };
                let right = pieces.remove(at + 1);
                pieces[at].push_str(&right);
            }
            for piece in pieces {
                result.push(
                    *self
                        .vocab
                        .get(&piece)
                        .ok_or(TokenizerError::InvalidVocabulary(piece))?,
                );
            }
        }
        Ok(())
    }
}

fn byte_alphabet() -> [char; 256] {
    let mut alphabet = ['\0'; 256];
    let mut visible = [false; 256];
    for byte in (b'!'..=b'~')
        .chain(b'\xA1'..=b'\xAC')
        .chain(b'\xAE'..=b'\xFF')
    {
        alphabet[byte as usize] = byte as char;
        visible[byte as usize] = true;
    }
    let mut next = 256;
    for byte in 0..=255usize {
        if !visible[byte] {
            alphabet[byte] = char::from_u32(next).expect("byte alphabet has valid Unicode values");
            next += 1;
        }
    }
    alphabet
}

#[cfg(test)]
mod tests {
    use super::ByteBpeTokenizer;

    #[test]
    fn rejects_invalid_gguf_vocabulary() {
        assert!(ByteBpeTokenizer::from_gguf_parts(
            vec!["a".into(), "a".into()],
            vec![],
            vec![1, 1]
        )
        .is_err());
        assert!(ByteBpeTokenizer::from_gguf_parts(vec!["a".into()], vec![], vec![]).is_err());
        assert!(ByteBpeTokenizer::from_gguf_parts(vec!["a".into()], vec![], vec![2]).is_err());
    }
}
