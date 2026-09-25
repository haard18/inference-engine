use std::fmt;

#[derive(Debug, PartialEq)]
pub enum EngineError {
    InvalidConfig(&'static str),
    InvalidShape {
        name: &'static str,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    InvalidValue(&'static str),
    InvalidToken(usize),
    EmptyPrompt,
    ContextFull,
}

impl fmt::Display for EngineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(message) => write!(f, "invalid model configuration: {message}"),
            Self::InvalidShape {
                name,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "invalid shape for {name}: expected {expected:?}, got {actual:?}"
                )
            }
            Self::InvalidValue(name) => write!(f, "invalid numeric value in {name}"),
            Self::InvalidToken(token) => write!(f, "token ID {token} is outside the vocabulary"),
            Self::EmptyPrompt => write!(f, "the prompt must contain at least one token"),
            Self::ContextFull => write!(f, "the model context is full"),
        }
    }
}

impl std::error::Error for EngineError {}
