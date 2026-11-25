//! Rust port of tiny-voca, a small Russian text-to-speech model.

pub mod fastmath;
pub mod model;
pub mod ops;
pub mod prof;
pub mod rng;
pub mod tts;
pub mod vocoder;
pub mod wav;
pub mod weights;

pub use model::{Config, Options, Synthesis, TinyVoca};
