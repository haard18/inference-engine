mod error;
mod gguf;
mod loader;
#[cfg(target_os = "macos")]
mod metal_backend;
#[cfg(target_os = "macos")]
pub use metal_backend::MetalRuntime;
mod model;
pub mod serving;
mod session;
mod tensor;
mod tokenizer;

pub use error::EngineError;
pub use gguf::{load_gguf, load_gguf_tokenizer, GgufError};
pub use loader::{load_safetensors, LoadError};
pub use model::{LayerWeights, Model, ModelConfig, ModelWeights};
pub use session::GenerationSession;
pub use tensor::Matrix;
pub use tokenizer::{ByteBpeDecoder, ByteBpeTokenizer, TokenizerError};
