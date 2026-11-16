//! Dizione: English text to phonemes. A Rust port of the Python package
//! <https://github.com/n4ze3m/dizione> that gives the same tokens.
//!
//! A dictionary built from CMUdict, a small model for words the dictionary
//! lacks, homograph rules, and text rules for numbers, dates, money, units and
//! chat forms.
//!
//! ```no_run
//! let phonemizer = dizione::Phonemizer::load(std::path::Path::new("weights/phonemizer")).unwrap();
//! println!("{}", phonemizer.phonemize("Hello, it costs $3.50.").unwrap());
//! ```

mod align;
mod convert;
mod engine;
mod g2p;
mod homographs;
mod morphology;
mod normalize;
mod numbers;
mod pyunicode;
mod rx;
mod sentence;
mod symbols;
mod tables;
mod unicode_tables;

pub use engine::Phonemizer;
pub use symbols::{Phonemized, SYMBOLS};

/// Token-convention version recorded by models trained on the output.
pub const VERSION: u32 = 4;
