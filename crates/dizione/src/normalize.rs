//! Text -> the words to say and the punctuation marks between them.
//!
//! Everything that is not a plain word is turned into words here, by rule:
//! numbers, ordinals, money, dates, times, telephone numbers, units,
//! abbreviations, initialisms, symbols, chat spellings, version numbers, codes
//! and reference numbers, web addresses and file names. Emoticons and emoji
//! are dropped. A pronunciation can be given inline, `{HH AH0 L OW1}` or
//! `{h ə l ˈoʊ}`, alone or right after the word it replaces.
//!
//! The patterns are those of the Python original, in the same order.

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use fancy_regex::Captures;
use unicode_normalization::char::canonical_combining_class;
use unicode_normalization::UnicodeNormalization;

use crate::convert::{convert, from_text, is_arpabet};
use crate::numbers::*;
use crate::pyunicode as py;
use crate::rx::{self, end, group, named, rx, start, sub, text as gtext};
use crate::symbols::{punctuation, Phones};
use crate::tables;

#[derive(Clone, Debug)]
pub struct Word {
    /// Letters and apostrophes, lower case.
    pub text: String,
    /// Continues the previous word (hyphenated compound, parts of one token).
    pub attach: bool,
    /// As written.
    pub raw: String,
    /// Read letter by letter.
    pub spell: bool,
    /// Pronunciation given inline by the user.
    pub phones: Option<Phones>,
}

impl Word {
    fn new(text: impl Into<String>, raw: impl Into<String>) -> Self {
        Word { text: text.into(), attach: false, raw: raw.into(), spell: false, phones: None }
    }
    fn spelled(text: impl Into<String>, raw: impl Into<String>) -> Self {
        Word { spell: true, ..Word::new(text, raw) }
    }
}

#[derive(Clone, Debug)]
pub enum Token {
    Word(Word),
    Punct(&'static str),
}

// Private-use characters that carry decisions through the rewriting of the
// text: letters to be named sit between SPELL and END, and an inline
// pronunciation is a single character (its number added to INLINE_BASE)
// between two INLINE.
const SPELL: char = '\u{E000}';
const END: char = '\u{E001}';
const INLINE: char = '\u{E003}';
const INLINE_BASE: u32 = 0xE100;

/// A set of words written as one string.
macro_rules! words {
    ($list:expr) => {{
        static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
        SET.get_or_init(|| $list.split_whitespace().collect())
    }};
}

const MONTHS: [&str; 12] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"];
const MONTH_PATTERN: &str = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec";
const DAY_PATTERN: &str = "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|Thurs|Thur|Tues|Mon|Tue|Wed|Thu|Fri|Sat|Sun";
const CURRENCY_CODES: &str = "USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY";
const ZONES: &str = "ET|CT|MT|PT|EST|EDT|CST|CDT|MST|MDT|PST|PDT|GMT|UTC|BST|CET|CEST|IST|AEST|AEDT|JST";
const SIZES: &str = "XXXL|XXL|XXS|XL|XS|S|M|L";
const LENGTHS: &str = "mm|cm|km|m|ft|in|inch|inches|feet|foot|metres?|meters?|yards?|yd|px|pixels?";
/// Before a number these mean a version: its parts are read as whole numbers ("three point twelve").
fn versioned(name: &str) -> bool {
    words!(
        "version ver release firmware patch kernel build python node nodejs ios ipados macos watchos tvos android \
         chrome firefox safari java php ruby rust swift kotlin golang django rails react angular vue docker \
         kubernetes ubuntu debian fedora windows linux postgres postgresql mysql redis numpy pytorch tensorflow cuda \
         xcode gcc clang unity minecraft typescript bluetooth usb hdmi opengl vulkan directx \
         upgrade upgraded upgrades update updated updates downgrade downgraded downgrades"
    )
    .contains(name.to_ascii_lowercase().as_str())
}

const HALF_DAY: &str = r"(?:\s?([AaPp])\.[Mm]\.?|\s?([AaPp])[Mm]\b)";
const AMOUNT: &str = r"(?P<whole>\d{1,3}(?:,\d{3})+|\d+)(?:\.(?P<fraction>\d+))?";
const CLOCK: &str = r"\d{1,2}:[0-5]\d(?:\s?[AaPp]\.?[Mm]\.?)?|\b\d{1,2}\s?[AaPp]\.?[Mm]\.?";

fn month_number(name: &str) -> Option<usize> {
    let key: String = name.to_lowercase().chars().take(3).collect();
    MONTHS.iter().position(|month| month.to_lowercase().starts_with(&key) && key.chars().count() == 3)
}

fn day_name(key: &str) -> Option<&'static str> {
    Some(match key {
        "mon" => "Monday",
        "tue" => "Tuesday",
        "wed" => "Wednesday",
        "thu" => "Thursday",
        "fri" => "Friday",
        "sat" => "Saturday",
        "sun" => "Sunday",
        _ => return None,
    })
}

type Coins = (&'static str, &'static str, &'static str, &'static str);

fn currency(symbol: &str) -> Option<Coins> {
    Some(match symbol {
        "$" | "USD" => ("dollar", "dollars", "cent", "cents"),
        "£" | "GBP" => ("pound", "pounds", "penny", "pence"),
        "€" | "EUR" => ("euro", "euros", "cent", "cents"),
        "¥" | "JPY" => ("yen", "yen", "", ""),
        "₹" | "INR" => ("rupee", "rupees", "paisa", "paise"),
        "CHF" => ("Swiss franc", "Swiss francs", "centime", "centimes"),
        "CAD" => ("Canadian dollar", "Canadian dollars", "cent", "cents"),
        "AUD" => ("Australian dollar", "Australian dollars", "cent", "cents"),
        "CNY" => ("yuan", "yuan", "", ""),
        _ => return None,
    })
}

fn symbol_words(c: char) -> Option<&'static str> {
    Some(match c {
        '&' => "and",
        '@' => "at",
        '+' => "plus",
        '=' => "equals",
        '%' => "percent",
        '×' => "times",
        '÷' => "divided by",
        '±' => "plus or minus",
        '©' => "copyright",
        '§' => "section",
        '°' => "degrees",
        '≥' => "greater than or equal to",
        '≤' => "less than or equal to",
        '≠' => "not equal to",
        '≈' => "approximately",
        '∞' => "infinity",
        '√' => "square root of",
        'π' => "pi",
        _ => return None,
    })
}

fn separator(piece: &str) -> &'static str {
    match piece {
        "." => "dot",
        "/" => "slash",
        "@" => "at",
        "-" => "dash",
        "_" => "underscore",
        ":" => "colon",
        "=" => "equals",
        "&" => "and",
        "+" => "plus",
        _ => "",
    }
}

fn roman(raw: &str) -> Option<u128> {
    Some(match raw {
        "II" => 2,
        "III" => 3,
        "IV" => 4,
        "VI" => 6,
        "VII" => 7,
        "VIII" => 8,
        "IX" => 9,
        "XI" => 11,
        "XII" => 12,
        "XIII" => 13,
        "XIV" => 14,
        "XV" => 15,
        "XVI" => 16,
        "XVIII" => 18,
        "XX" => 20,
        _ => return None,
    })
}

// Words whose accent is the only sign of how they are read.
fn accent_decides(word: &str) -> Option<&'static str> {
    Some(match word {
        "résumé" | "resumé" => "R EH1 Z AH0 M EY2",
        "exposé" => "EH2 K S P OW0 Z EY1",
        "rosé" => "R OW0 Z EY1",
        "pâté" => "P AA0 T EY1",
        "maté" => "M AA1 T EY2",
        "saké" => "S AA1 K EY2",
        "lamé" => "L AA0 M EY1",
        _ => return None,
    })
}

/// An accented Latin letter without its accent; other characters are left alone.
fn plain_letter(c: char, out: &mut String) {
    if c.is_ascii() || !py::is_alpha(c) {
        out.push(c);
        return;
    }
    let form = match c {
        'ß' => "ss",
        'æ' => "ae",
        'Æ' => "Ae",
        'œ' => "oe",
        'Œ' => "Oe",
        'ø' => "o",
        'Ø' => "O",
        'đ' => "d",
        'ł' => "l",
        'Ł' => "L",
        _ => "",
    };
    if !form.is_empty() {
        out.push_str(form);
        return;
    }
    let base: String = c.nfkd().filter(|d| canonical_combining_class(*d) == 0).collect();
    if !base.is_empty() && base.chars().all(|d| d.is_ascii_alphabetic()) {
        out.push_str(&base);
    } else {
        out.push(c);
    }
}

fn spelled(letters: &str) -> String {
    format!("{SPELL}{letters}{END}")
}

fn ascii_letter(c: char) -> bool {
    c.is_ascii_alphabetic()
}

/// Maximal runs of ASCII letters (`re.findall(r"[A-Za-z]+", text)`).
fn letter_runs(text: &str) -> Vec<&str> {
    text.split(|c: char| !ascii_letter(c)).filter(|run| !run.is_empty()).collect()
}

/// The run of ASCII letters at the start of `text`, after optional spaces
/// (`re.match(r"\s*([A-Za-z]+)", text)`).
fn word_after(text: &str) -> Option<&str> {
    let rest = py::lstrip(text);
    let length = rest.find(|c: char| !ascii_letter(c)).unwrap_or(rest.len());
    if length > 0 { Some(&rest[..length]) } else { None }
}

/// `re.match(r"[A-Za-z']+", text)`.
fn first_word(text: &str) -> Option<&str> {
    let length = text.find(|c: char| !(ascii_letter(c) || c == '\'')).unwrap_or(text.len());
    if length > 0 { Some(&text[..length]) } else { None }
}

/// True when the text after an abbreviation's full stop starts a new sentence
/// (or there is none): the stop is then kept as a sentence end.
fn sentence_follows(following: &str) -> bool {
    match py::first(following) {
        None => true,
        Some(c) => c.is_uppercase() && first_word(following).is_some_and(|word| words!(SENTENCE_STARTERS).contains(word.to_lowercase().as_str())),
    }
}

const SENTENCE_STARTERS: &str = "the he she it i we they this that there but and in on at a an if when what why how who my our \
you then so now after before however";

/// The words just before `position` in the same clause, in lower case.
fn before(text: &str, position: usize, count: usize) -> Vec<String> {
    let window = py::tail(&text[..position], 80);
    // re.split(r"[.!?;…]\s|[()\"]", window)[-1]
    let mut clause_start = 0;
    let mut chars = window.char_indices().peekable();
    while let Some((at, c)) = chars.next() {
        if matches!(c, '.' | '!' | '?' | ';' | '…') {
            if let Some(&(next_at, next)) = chars.peek() {
                if py::is_space(next) {
                    chars.next();
                    clause_start = next_at + next.len_utf8();
                }
            }
        } else if matches!(c, '(' | ')' | '"') {
            clause_start = at + c.len_utf8();
        }
    }
    let found = letter_runs(&window[clause_start..]);
    found[found.len().saturating_sub(count)..].iter().map(|word| word.to_lowercase()).collect()
}

/// What the words before a number say it is: `"phone"`, `"pin"`, `"strong"`
/// (a number, an ID, a reference), `"weak"` (an order, a ticket), `"enter"`.
fn identifier(text: &str, position: usize) -> Option<&'static str> {
    let phone = words!("call dial ring phone telephone tel mobile cell fax hotline text sms whatsapp");
    let found = before(text, position, 4);
    let last = found.last()?.as_str();
    let n = found.len();
    if phone.contains(last)
        || (words!("to on at or us me him her them back").contains(last) && found[n.saturating_sub(3)..n - 1].iter().any(|word| phone.contains(word.as_str())))
    {
        return Some("phone");
    }
    let mut index = n - 1;
    while index > 0 && words!("is was are were ends ending in with starts starting begins beginning").contains(found[index].as_str()) {
        index -= 1;
    }
    let word = found[index].as_str();
    if phone.contains(word) {
        return Some("phone");
    }
    if words!("code pin extension ext passcode password zip postcode cvv cvc otp").contains(word) {
        return Some("pin");
    }
    if words!("number id reference ref serial account tracking booking confirmation").contains(word) {
        return Some("strong");
    }
    if words!("order ticket invoice case claim policy receipt transaction parcel shipment container payment \
               reservation membership customer batch lot item job issue bug request")
        .contains(word)
    {
        return Some("weak");
    }
    if last == "enter" { Some("enter") } else { None }
}

/// Digits inside a file name or an address: 3, 2026, 0412.
fn address_digits(text: &str) -> String {
    let length = text.chars().count();
    if length <= 2 && !(length == 2 && text.starts_with('0')) {
        return cardinal_of(text);
    }
    if length == 4 && (1900..=2099).contains(&py::int(text)) {
        return year(py::int(text) as u128);
    }
    digits(text, "zero")
}

/// `re.escape`.
fn escape(text: &str) -> String {
    let mut out = String::new();
    for c in text.chars() {
        if "()[]{}?*+-|^$\\.&~# \t\n\r\x0b\x0c".contains(c) {
            out.push('\\');
        }
        out.push(c);
    }
    out
}

/// Pieces of a run of letters and digits: `re.findall(r"[A-Za-z]+|\d+", text)`.
fn letter_digit_runs(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut run_start: Option<(usize, bool)> = None;
    for (at, c) in text.char_indices() {
        let kind = if ascii_letter(c) { Some(true) } else if py::is_decimal(c) { Some(false) } else { None };
        match (run_start, kind) {
            (Some((_, k)), Some(kind)) if k == kind => {}
            (Some((from, _)), _) => {
                out.push(&text[from..at]);
                run_start = kind.map(|k| (at, k));
            }
            (None, Some(kind)) => run_start = Some((at, kind)),
            (None, None) => {}
        }
    }
    if let Some((from, _)) = run_start {
        out.push(&text[from..]);
    }
    out
}

fn has_currency_code(text: &str) -> bool {
    CURRENCY_CODES.split('|').any(|code| text.contains(code))
}

/// True when `text` holds the first three letters of a month name, capital first.
fn has_month(text: &str) -> bool {
    text.as_bytes().windows(3).any(|w| {
        matches!(w[0], b'J' | b'F' | b'M' | b'A' | b'S' | b'O' | b'N' | b'D')
            && matches!(w, b"Jan" | b"Feb" | b"Mar" | b"Apr" | b"May" | b"Jun" | b"Jul" | b"Aug" | b"Sep" | b"Oct" | b"Nov" | b"Dec")
    })
}

/// True when a full stop stands between two digits somewhere in `text`.
fn has_decimal_point(text: &str) -> bool {
    let mut before = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '.' && before && chars.peek().is_some_and(|next| py::is_decimal(*next)) {
            return true;
        }
        before = py::is_decimal(c);
    }
    false
}

/// False when `text` cannot hold a web address, a mail address or a file name.
fn may_hold_address(text: &str) -> bool {
    let bytes = text.as_bytes();
    text.contains('@')
        || text.contains("://")
        || text.contains("www.")
        || bytes.windows(2).any(|pair| pair[0] == b'.' && pair[1].is_ascii_alphabetic())
}

/// Both cases of every ASCII letter of a pattern made of literals: `km` -> `[kK][mM]`.
fn either_case(pattern: &str) -> String {
    let mut out = String::with_capacity(pattern.len() * 4);
    let mut escaped = false;
    for c in pattern.chars() {
        if c.is_ascii_alphabetic() && !escaped {
            out.push('[');
            out.push(c.to_ascii_lowercase());
            out.push(c.to_ascii_uppercase());
            out.push(']');
        } else {
            out.push(c);
        }
        escaped = c == '\\' && !escaped;
    }
    out
}

pub struct Normalizer {
    abbreviations: HashMap<&'static str, (&'static str, &'static str)>,
    units: HashMap<String, (&'static str, &'static str)>,
    exact_units: HashMap<&'static str, (&'static str, &'static str)>,
    chat: HashMap<&'static str, &'static str>,
    endings: HashMap<&'static str, &'static str>,
    /// The unit names as one alternation, longest first, for a pattern that ignores case.
    unit_names: String,
    /// The same with both cases of every ASCII letter written out. Text without
    /// letters beyond ASCII needs no other case folding, and the pattern is much
    /// quicker to build than one that ignores case throughout.
    unit_names_ascii: String,
}

impl Normalizer {
    pub fn new() -> Self {
        let abbreviations = tables::rows(tables::ABBREVIATIONS).into_iter().filter(|(_, c)| c.len() > 2).map(|(_, c)| (c[0], (c[1], c[2]))).collect();
        // A unit marked "exact" is written in one letter case only: "m" is a meter, "M" is not.
        let (mut units, mut exact_units) = (HashMap::new(), HashMap::new());
        for (_, c) in tables::rows(tables::UNITS) {
            if c.len() < 3 {
                continue;
            }
            if c.len() > 3 && c[3] == "exact" {
                exact_units.insert(c[0], (c[1], c[2]));
            } else {
                units.insert(c[0].to_lowercase(), (c[1], c[2]));
            }
        }
        let chat = tables::rows(tables::CHAT).into_iter().map(|(_, c)| (c[0], if c.len() > 1 { c[1] } else { "-" })).collect();
        let mut names: Vec<String> = units.keys().map(|unit| escape(unit)).chain(exact_units.keys().map(|unit| escape(unit))).collect::<HashSet<_>>().into_iter().collect();
        names.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()).then_with(|| a.cmp(b)));
        let unit_names = names.join("|");
        let unit_names_ascii = either_case(&unit_names);
        Normalizer { abbreviations, units, exact_units, chat, endings: tables::mapping(tables::ENDINGS), unit_names, unit_names_ascii }
    }

    /// `re.match(r"\s?(?:<others><units>)(?![A-Za-z])", ..., re.IGNORECASE)`, for the text after a number.
    fn unit_ahead(&self, wide: bool, others: &str) -> String {
        if wide {
            format!(r"(?i)^\s?(?:{}{})(?![A-Za-z])", others, self.unit_names)
        } else {
            format!(r"^\s?(?:{}{})(?![A-Za-z])", either_case(others), self.unit_names_ascii)
        }
    }

    /// Words and punctuation marks of `text`. `is_word(spelling)` tells whether
    /// the dictionary knows a spelling.
    pub fn tokens(&self, text: &str, is_word: &dyn Fn(&str) -> bool) -> Result<Vec<Token>, String> {
        let mut run = Run { n: self, is_word, inline: Vec::new(), day_first: false, dated: false, error: None };
        let out = run.tokens(text);
        match run.error {
            Some(error) => Err(error),
            None => out,
        }
    }
}

/// One text on its way through the rules.
struct Run<'a> {
    n: &'a Normalizer,
    is_word: &'a dyn Fn(&str) -> bool,
    inline: Vec<(String, Phones)>,
    day_first: bool,
    dated: bool,
    error: Option<String>,
}

impl Run<'_> {
    fn known(&self, word: &str) -> bool {
        (self.is_word)(word)
    }

    // -- pieces of the text that are not words --

    fn amount(&self, symbol: &str, whole: &str, fraction: Option<&str>, scale: Option<&str>) -> String {
        let (one, many, cent, cents) = currency(symbol).unwrap_or(("", "", "", ""));
        let fraction = fraction.filter(|f| !f.is_empty());
        if let Some(scale) = scale.filter(|s| !s.is_empty()) {
            let lower = scale.to_lowercase();
            let scale = match lower.as_str() {
                "k" => "thousand",
                "m" => "million",
                "b" | "bn" => "billion",
                "t" => "trillion",
                other => other,
            };
            let number = match fraction {
                Some(fraction) => format!("{whole}.{fraction}"),
                None => whole.to_string(),
            };
            return format!("{} {} {}", number_words(&number), scale, many);
        }
        if let Some(fraction) = fraction {
            if fraction.chars().count() > 2 || cent.is_empty() {
                return format!("{} {}", number_words(&format!("{whole}.{fraction}")), many);
            }
        }
        let amount = value_of(whole);
        let mut change_text = fraction.unwrap_or("0").to_string();
        while change_text.chars().count() < 2 {
            change_text.push('0');
        }
        let change = py::int(&change_text);
        let mut parts = Vec::new();
        if amount != Some(0) || change == 0 {
            parts.push(format!("{} {}", cardinal_of(whole), if amount == Some(1) { one } else { many }));
        }
        if change != 0 && !cent.is_empty() {
            parts.push(format!("{} {}", cardinal(change as u128), if change == 1 { cent } else { cents }));
        }
        parts.join(" and ")
    }

    fn money(&self, c: &Captures) -> String {
        let whole = named(c, "whole").unwrap_or("").replace(',', "");
        let said = self.amount(named(c, "symbol").unwrap_or(""), &whole, named(c, "fraction"), named(c, "scale"));
        let minus = named(c, "minus").is_some_and(|m| !m.is_empty()) || named(c, "inner").is_some_and(|m| !m.is_empty());
        format!("{}{} ", if minus { " minus " } else { " " }, said)
    }

    /// " A M" or " P M" as letters. A full stop that closes "p.m." is kept as
    /// a sentence end only when a new sentence follows.
    fn half_day(text: &str, c: &Captures, dotted: Option<&str>, plain: Option<&str>) -> String {
        let Some(letter) = dotted.or(plain).filter(|l| !l.is_empty()) else { return String::new() };
        let following = py::lstrip(&text[end(c)..]);
        let closes = dotted.is_some_and(|d| !d.is_empty()) && gtext(c, 0).ends_with('.') && sentence_follows(following);
        format!("{SPELL}{}M{END}{} ", letter.to_uppercase(), if closes { " ." } else { "" })
    }

    fn time(&self, text: &str, c: &Captures) -> String {
        let (hour, minute) = (py::int(gtext(c, 1)) as u128, py::int(gtext(c, 2)) as u128);
        let second = group(c, 3);
        let half = group(c, 4).or(group(c, 5));
        let earlier = before(text, start(c), 3);
        if let Some(second) = second {
            if half.is_none() && earlier.last().map(String::as_str) != Some("at") {
                return format!(" {} ", duration(hour, minute, py::int(second) as u128));
            }
        }
        if hour > 24 && second.is_none() && half.is_none() {
            let lengths = words!("lasted took duration runtime recording video clip track lap pace length long");
            let after = word_after(&text[end(c)..]);
            if earlier.iter().any(|word| lengths.contains(word.as_str())) || after.is_some_and(|word| lengths.contains(word.to_lowercase().as_str())) {
                return format!(" {} ", duration(0, hour, minute));
            }
            return format!(" {} {}{} ", cardinal(hour), if minute < 10 { "oh " } else { "" }, cardinal(minute));
        }
        if hour > 24 {
            return gtext(c, 0).to_string();
        }
        let mut said = cardinal(hour);
        if minute == 0 {
            if half.is_some() {
            } else if hour == 0 {
                said = "midnight".to_string();
            } else if hour > 12 {
                said.push_str(" hundred");
            } else {
                said.push_str(" o'clock");
            }
        } else {
            said.push_str(if minute < 10 { " oh " } else { " " });
            said.push_str(&cardinal(minute));
        }
        if let Some(second) = second {
            let seconds = py::int(second);
            said.push_str(&format!(" and {}{}", cardinal(seconds as u128), if seconds == 1 { " second" } else { " seconds" }));
        }
        format!(" {} {}", said, Self::half_day(text, c, group(c, 4), group(c, 5)))
    }

    /// "9.30 to 11.30" where the words before say it is a time of day.
    fn dot_times(&self, text: &str, c: &Captures) -> String {
        let following = &text[end(c)..];
        let clock = words!("window slot open opens opening hours shift session served service available booked appointment \
                            delivery arrive arrives arriving");
        let unit = rx!(following, self.n.unit_ahead(rx::is_wide(following), "%|percent|million|billion|"));
        if py::int(gtext(c, 1)) > 23
            || py::int(gtext(c, 4)) > 23
            || !before(text, start(c), 4).iter().any(|word| clock.contains(word.as_str()))
            || rx::found(unit, following)
        {
            return gtext(c, 0).to_string();
        }
        format!("{}:{}{}{}:{}", gtext(c, 1), gtext(c, 2), gtext(c, 3), gtext(c, 4), gtext(c, 5))
    }

    /// 16:9, 3:1, 1:25,000. Two digits after the colon are a time unless the words say "ratio".
    fn ratio(&self, text: &str, c: &Captures) -> String {
        let (first, second) = (gtext(c, 1), gtext(c, 2));
        if first.chars().count() <= 2 && second.chars().count() == 2 && matches!(py::first(second), Some('0'..='5')) {
            let ratios = words!("ratio ratios scale odds aspect mix split proportion");
            let after = word_after(&text[end(c)..]);
            if !(before(text, start(c), 3).iter().any(|word| ratios.contains(word.as_str())) || after.is_some_and(|word| ratios.contains(word.to_lowercase().as_str()))) {
                return gtext(c, 0).to_string();
            }
        }
        format!(" {} to {} ", number_words(first), number_words(second))
    }

    fn date(month: u64, day: u64, year_text: Option<&str>, the: bool) -> Option<String> {
        if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
            return None;
        }
        let name = MONTHS[month as usize - 1];
        let mut said = if the { format!("the {} of {}", ordinal(day as u128), name) } else { format!("{} {}", name, ordinal(day as u128)) };
        if let Some(year_text) = year_text.filter(|y| !y.is_empty()) {
            let number = py::int(year_text) as u128;
            said.push(' ');
            if year_text.chars().count() == 4 {
                said.push_str(&year(number));
            } else {
                said.push_str(&format!("{}{}", if number < 10 { "oh " } else { "" }, cardinal(number)));
            }
        }
        Some(format!(" {said} "))
    }

    fn slash_date(&self, c: &Captures) -> String {
        let (mut first, mut second) = (py::int(gtext(c, 1)), py::int(gtext(c, 2)));
        if (first > 12 && 12 >= second) || (self.day_first && second <= 12) {
            std::mem::swap(&mut first, &mut second);
        }
        Self::date(first, second, group(c, 3), false).unwrap_or_else(|| gtext(c, 0).to_string())
    }

    /// 15.11.2026: dates written with full stops put the day first.
    fn dot_date(c: &Captures) -> String {
        let (mut day, mut month) = (py::int(gtext(c, 1)), py::int(gtext(c, 2)));
        if month > 12 && 12 >= day {
            std::mem::swap(&mut day, &mut month);
        }
        Self::date(month, day, group(c, 3), false).unwrap_or_else(|| gtext(c, 0).to_string())
    }

    /// "22 February", "5 Jan. 2026".
    fn day_month(c: &Captures) -> String {
        let month = gtext(c, 2);
        match Self::date(month_number(month).map_or(0, |m| m as u64 + 1), py::int(gtext(c, 1)), group(c, 4), true) {
            None => gtext(c, 0).to_string(),
            Some(said) => format!("{}{}", said, if !gtext(c, 3).is_empty() && MONTHS.contains(&month) { ". " } else { "" }),
        }
    }

    /// "3-17 November" and "November 3-17".
    fn days_of_month(c: &Captures) -> String {
        let month_first = py::first(gtext(c, 1)).is_some_and(py::is_alpha);
        let (month, first, second) = if month_first { (gtext(c, 1), gtext(c, 2), gtext(c, 3)) } else { (gtext(c, 3), gtext(c, 1), gtext(c, 2)) };
        let (first, second) = (py::int(first), py::int(second));
        if !((1..=31).contains(&first) && (1..=31).contains(&second)) {
            return gtext(c, 0).to_string();
        }
        let name = MONTHS[month_number(month).unwrap_or(0)];
        let said = if month_first {
            format!("{} {} to {}", name, ordinal(first as u128), ordinal(second as u128))
        } else {
            format!("the {} to the {} of {}", ordinal(first as u128), ordinal(second as u128), name)
        };
        let year_said = group(c, 4).filter(|y| !y.is_empty()).map(|y| format!(" {}", year(py::int(y) as u128))).unwrap_or_default();
        format!(" {said}{year_said} ")
    }

    /// "Mon.-Fri.", "Jan-Mar": from one day or month to another.
    fn day_range(&self, text: &str, c: &Captures) -> String {
        let name = |name: &str| -> String {
            let key = name.to_lowercase();
            match self.n.abbreviations.get(key.as_str()) {
                Some((_, said)) => said.to_string(),
                None => day_name(&key).unwrap_or(name).to_string(),
            }
        };
        let following = py::lstrip(&text[end(c)..]);
        let ends = !gtext(c, 4).is_empty() && sentence_follows(following);
        format!("{} to {}{}", name(gtext(c, 1)), name(gtext(c, 3)), if ends { " ." } else { "" })
    }

    /// "14/03", "12/10" as a day of the year, or `None` when it is a fraction.
    fn short_date(&self, text: &str, c: &Captures) -> Option<String> {
        let (first_text, second_text) = (gtext(c, 1), gtext(c, 2));
        let (first_length, second_length) = (first_text.chars().count(), second_text.chars().count());
        if first_length > 2 || second_length > 2 {
            return None;
        }
        let (mut first, mut second) = (py::int(first_text), py::int(second_text));
        let earlier = before(text, start(c), 4);
        if (1..=12).contains(&first) && (20..=49).contains(&second) && earlier.iter().any(|word| word == "card") {
            return Some(format!(" {} {} ", MONTHS[first as usize - 1], year(2000 + second as u128)));
        }
        let last = earlier.last().map(String::as_str).unwrap_or("");
        let classic = first < second && matches!(second, 2 | 3 | 4 | 5 | 6 | 8 | 10 | 16 | 32 | 64 | 100);
        let zero = (first_length == 2 && first_text.starts_with('0')) || (second_length == 2 && second_text.starts_with('0'));
        let date_before = words!("on dated due until till by since from before after for shipped delivered arriving arrives arrive \
                                  arrived expires expired ends ending starts starting begins beginning monday tuesday wednesday \
                                  thursday friday saturday sunday mon tue tues wed thu thur thurs fri sat sun");
        if !(zero
            || (12 < first && first <= 31 && second <= 12 && (first, second) != (24, 7))
            || (date_before.contains(last) && (!classic || last == "on"))
            || (self.dated && !classic && first != second))
        {
            return None;
        }
        if first > 12 || (self.day_first && second <= 12) {
            std::mem::swap(&mut first, &mut second);
        }
        Self::date(first, second, None, false)
    }

    fn fraction(&self, text: &str, c: &Captures) -> String {
        if let Some(said) = self.short_date(text, c) {
            return said;
        }
        let (top_text, bottom_text) = (gtext(c, 1), gtext(c, 2));
        let (top, bottom) = (value_of(top_text), value_of(bottom_text));
        let same = match (top, bottom) {
            (Some(a), Some(b)) => a == b,
            (None, None) => py::ascii_digits(top_text).trim_start_matches('0') == py::ascii_digits(bottom_text).trim_start_matches('0'),
            _ => false,
        };
        if (top, bottom) == (Some(24), Some(7)) || same {
            return format!(" {} {} ", cardinal_of(top_text), cardinal_of(bottom_text));
        }
        match (top, bottom) {
            (Some(top), Some(bottom)) if 0 < top && top < 100 && 1 < bottom && bottom <= 100 => format!(" {} ", fraction_words(top, bottom, false)),
            _ => format!(" {} over {} ", cardinal_of(top_text), cardinal_of(bottom_text)),
        }
    }

    /// "2 1/2": two and a half.
    fn mixed_fraction(c: &Captures) -> String {
        let (top, bottom) = (py::int(gtext(c, 2)) as u128, py::int(gtext(c, 3)) as u128);
        if !(0 < top && top < bottom) {
            return gtext(c, 0).to_string();
        }
        format!(" {} and {} ", number_words(gtext(c, 1)), fraction_words(top, bottom, true))
    }

    /// "½" and "1½". Runs before the text is normalised.
    fn vulgar(c: &Captures) -> String {
        let normal: String = gtext(c, 2).nfkc().collect();
        let mut parts = normal.split('\u{2044}');
        let top = py::int(parts.next().unwrap_or(""));
        let bottom = py::int(parts.next().unwrap_or(""));
        match group(c, 1) {
            None => format!(" {top}/{bottom} "),
            Some(whole) => format!(" {} and {} ", cardinal_of(whole), fraction_words(top as u128, bottom as u128, true)),
        }
    }

    fn phone(c: &Captures) -> String {
        let said: Vec<String> = (1..c.len())
            .filter_map(|index| group(c, index))
            .filter(|g| !g.is_empty())
            .map(|g| if g == "800" { "eight hundred".to_string() } else { digits(g, "oh") })
            .collect();
        format!(" {} ", said.join(", "))
    }

    fn measure(&self, text: &str, c: &Captures) -> String {
        let unit = gtext(c, 3);
        let names = self.n.exact_units.get(unit).or_else(|| self.n.units.get(unit.to_lowercase().as_str()));
        let unit_end = c.get(3).map_or(0, |m| m.end());
        let next = py::first(&text[unit_end..]);
        let Some(&(one, many)) = names else { return gtext(c, 0).to_string() };
        if unit.chars().count() == 1 && matches!(next, Some('-') | Some('\'') | Some('/')) {
            return gtext(c, 0).to_string();
        }
        let value = gtext(c, 1);
        if let Some(upper) = group(c, 2).filter(|u| !u.is_empty()) {
            return format!(" {} to {} {} ", number_words(value), number_words(upper), many);
        }
        format!(" {} {} ", number_words(value), if matches!(value, "1" | "1.0") { one } else { many })
    }

    /// "120 m²", "3 m3": square and cubic measures.
    fn square(&self, c: &Captures) -> String {
        let unit = gtext(c, 2);
        let names = if unit == "in" { Some(("inch", "inches")) } else { self.n.exact_units.get(unit).or_else(|| self.n.units.get(unit)).copied() };
        let Some((one, many)) = names else { return gtext(c, 0).to_string() };
        let kind = if matches!(gtext(c, 3), "²" | "2") { "square" } else { "cubic" };
        format!(" {} {} {} ", number_words(gtext(c, 1)), kind, if matches!(gtext(c, 1), "1" | "1.0") { one } else { many })
    }

    /// True when a plural noun follows a number: it is then an amount, not a reference.
    fn counted(&self, text: &str, at: usize) -> bool {
        let Some(after) = word_after(&text[at..]) else { return false };
        let word = after.to_lowercase();
        if matches!(word.as_str(), "people" | "men" | "women" | "children" | "feet" | "miles") {
            return true;
        }
        word.len() > 3
            && word.ends_with('s')
            && !word.ends_with("ss")
            && !words!("is was has does goes ships arrives departs leaves shows needs blocks remains includes contains \
                        says reads starts ends closes opens runs takes comes this his its us as yes plus minus less")
                .contains(word.as_str())
            && self.known(&word[..word.len() - 1])
    }

    fn number(&self, text: &str, c: &Captures) -> String {
        let value = gtext(c, 0);
        let plain = value.replace(',', "");
        let length = plain.chars().count();
        if !plain.contains('.') && !value.contains(',') && !plain.starts_with('0') && length > 2 {
            let kind = identifier(text, start(c));
            let limit = match kind {
                Some("phone") | Some("pin") => 3,
                Some("strong") | Some("enter") => 4,
                Some("weak") => 5,
                _ => 7,
            };
            if length >= limit
                && !(kind.is_none() && plain.ends_with("000"))
                && !(kind == Some("enter") && length == 4 && (1900..=2099).contains(&py::int(&plain)))
                && !(matches!(kind, None | Some("phone") | Some("weak")) && self.counted(text, end(c)))
            {
                return format!(" {} ", digits(&plain, if kind == Some("phone") { "oh" } else { "zero" }));
            }
        }
        if !plain.contains('.') && !value.contains(',') && length == 4 && !plain.starts_with('0') {
            let number = py::int(&plain) as u128;
            let head = &text[..start(c)];
            let earlier = letter_runs(py::tail(head, 24));
            let after = word_after(&text[end(c)..]).map(|word| word.to_lowercase());
            let counted = after.as_deref().is_some_and(|word| {
                let b = word.as_bytes();
                (b.len() >= 2 && b[b.len() - 1] == b's' && b[b.len() - 2] != b's') || ["people", "men", "women", "children", "feet", "miles"].iter().any(|w| word.ends_with(w))
            });
            let dated = earlier.last().is_some_and(|last| {
                let lower = last.to_lowercase();
                words!("in since by from until till before after during around circa of year early late mid summer winter \
                        spring fall autumn class")
                    .contains(lower.as_str())
                    || (lower.len() >= 3 && month_number(&lower).is_some())
            });
            // re.search(r"(?<![A-Za-z])([A-Z]{2,5}) $", head)
            let model = head.strip_suffix('\n').unwrap_or(head).strip_suffix(' ').or_else(|| head.strip_suffix(' ')).and_then(|body| {
                let run = body.len() - body.trim_end_matches(|ch: char| ch.is_ascii_uppercase()).len();
                let preceded = body[..body.len() - run].chars().next_back().is_some_and(ascii_letter);
                if (2..=5).contains(&run) && !preceded { Some(&body[body.len() - run..]) } else { None }
            });
            if let Some(model) = model {
                if !words!("IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR BUT NOT ALL").contains(model) && !counted {
                    return format!(" {} ", pairs(number));
                }
            }
            if (1100..=1999).contains(&number) || ((2000..=2099).contains(&number) && (dated || !counted)) {
                return format!(" {} ", year(number));
            }
        }
        format!(" {} ", number_words(value))
    }

    // -- codes and reference numbers --

    /// Letters inside a code that are a word to say, not letters to name: SAVE-20.
    fn reads_as_word(&self, letters: &str) -> bool {
        let lower = letters.to_lowercase();
        let length = lower.chars().count();
        if length < 3 || !self.known(&lower) {
            return false;
        }
        length > 3 || ["s", "es", "ed", "ing"].iter().any(|ending| self.known(&format!("{lower}{ending}")))
    }

    /// A group that mixes letters and digits: the letters are named, the digits read one by one.
    fn alnum(group: &str, zero: &str) -> String {
        letter_digit_runs(group).iter().map(|run| if py::all_digit(run) { digits(run, zero) } else { spelled(run) }).collect::<Vec<_>>().join(" ")
    }

    /// Groups of digits with a short break between them. Small numbers stay numbers: 9-1-1.
    fn digit_groups(groups: &[&str], zero: &str) -> String {
        if groups.iter().all(|g| short_number(g) && g.chars().count() <= 2) {
            return groups.iter().map(|g| cardinal_of(g)).collect::<Vec<_>>().join(" , ");
        }
        groups.iter().map(|g| digits(g, zero)).collect::<Vec<_>>().join(" , ")
    }

    fn code_words(&self, groups: &[&str], text: &str, c: &Captures) -> Option<String> {
        let token = groups.join("-");
        if !token.chars().any(py::is_digit) {
            return None; // well-known, Wi-Fi
        }
        let kinds: Vec<char> = groups
            .iter()
            .map(|g| if py::all_digit(g) { 'd' } else if g.chars().any(py::is_digit) { 'm' } else if py::is_upper(g) { 'u' } else { 'l' })
            .collect();
        if kinds.iter().all(|k| *k == 'd') {
            if groups.len() == 2 {
                let years = groups.iter().all(|g| g.len() == 4 && (1000..=2100).contains(&py::int(g)));
                if !(!gtext(c, 1).is_empty()
                    || groups.iter().any(|g| g.len() > 1 && g.starts_with('0'))
                    || (groups.iter().map(|g| g.len()).min().unwrap_or(0) >= 4 && !years)
                    || matches!(identifier(text, start(c)), Some("strong") | Some("pin")))
                {
                    return None; // 10-15 and 2019-2021 are ranges
                }
            }
            return Some(Self::digit_groups(groups, "zero"));
        }
        if groups.len() == 2 && kinds[1] == 'd' && matches!(kinds[0], 'u' | 'l') {
            let (letters, number) = (groups[0], groups[1]);
            if (kinds[0] == 'l' && letters.len() > 1) || self.reads_as_word(letters) {
                return None; // pre-2020, COVID-19, SAVE-20
            }
            if short_number(number) {
                return Some(format!("{} {}", spelled(letters), cardinal_of(number))); // B-12, KX-300
            }
            return Some(format!("{} , {}", spelled(letters), digits(number, "zero")));
        }
        let digits_then_letters = |g: &str| {
            let letters = g.trim_start_matches(|ch: char| ch.is_ascii_digit());
            letters.len() < g.len() && !letters.is_empty() && letters.chars().all(ascii_letter)
        };
        if groups.len() == 2 && kinds[0] == 'd' && (kinds[1] == 'l' || groups[1].len() == 1 || digits_then_letters(groups[1])) {
            return None; // 5-year, 3-D, 5-10MB
        }
        let counted = |g: &str| {
            let rest = g.trim_start_matches(|ch: char| ch.is_ascii_digit());
            rest.len() < g.len() && matches!(rest, "s" | "st" | "nd" | "rd" | "th")
        };
        if kinds.iter().zip(groups).any(|(kind, g)| (*kind == 'l' && g.len() > 2) || (*kind == 'm' && counted(g))) {
            return None; // 20-year-old, F-16s
        }
        if !token.chars().any(|ch| ch.is_uppercase()) && kinds.iter().any(|k| matches!(k, 'l' | 'm')) {
            return None; // 3-in-1, 50s-60s
        }
        let mut parts = Vec::new();
        for (kind, group) in kinds.iter().zip(groups) {
            parts.push(match kind {
                'd' => digits(group, "zero"),
                'm' => Self::alnum(group, "zero"),
                _ if group.len() > 3 && self.known(&group.to_lowercase()) => group.to_string(),
                _ => if group.len() <= 5 { spelled(group) } else { group.to_string() },
            });
        }
        Some(parts.join(" , "))
    }

    /// Codes written with hyphens: AB-4471-Z, 91-220-0017, #A7-4419.
    fn code(&self, text: &str, c: &Captures) -> String {
        let groups: Vec<&str> = gtext(c, 2).split('-').collect();
        match self.code_words(&groups, text, c) {
            None => gtext(c, 0).to_string(),
            Some(said) => format!("{}{} ", if !gtext(c, 1).is_empty() { " number " } else { " " }, said),
        }
    }

    /// Digits written in groups with spaces: a telephone number, a card or
    /// tracking number, or a large number with spaces between the thousands.
    fn grouped(&self, text: &str, c: &Captures) -> String {
        let plus = named(c, "plus").is_some();
        let mut lead = named(c, "lead");
        let mut trail = named(c, "trail");
        let core: Vec<&str> = named(c, "core1").or(named(c, "core2")).unwrap_or("").split(' ').collect();
        let (mut head, mut tail) = (String::new(), String::new());
        if let Some(found) = lead {
            if words!("IN ON AT TO OF BY OR IS AS IF NO SO WE HE MY ME DO GO UP AN AM PM BE IT US THE AND FOR BUT NOT ALL").contains(found) {
                // "IN 2019 150 ..." in capitals
                head = format!("{found} ");
                tail = trail.map(|t| format!(" {t}")).unwrap_or_default();
                lead = None;
                trail = None;
            }
        }
        let kind = identifier(text, start(c));
        let pure = core.iter().all(|part| py::all_digit(part));
        if pure && !plus && lead.is_none() {
            let following = &text[end(c)..];
            let unit = rx!(following, self.n.unit_ahead(rx::is_wide(following), ""));
            let amount = self.counted(text, end(c)) || rx::found(unit, following);
            if kind.is_none()
                && !core[0].starts_with('0')
                && core[0].chars().count() <= 3
                && core[1..].iter().all(|part| part.chars().count() == 3)
                && (*core.last().unwrap() == "000" || (amount && core.len() > 2))
            {
                return format!("{}{}{}", head, core.join(","), tail); // 1 000 000: the number rules read it
            }
            if !(kind.is_some() || (core[0].chars().count() > 1 && core[0].starts_with('0')) || (core.len() > 2 && core.iter().any(|part| part.chars().count() > 2))) {
                return gtext(c, 0).to_string();
            }
        } else if !pure && lead.is_none() && !plus && kind.is_none() {
            return gtext(c, 0).to_string();
        }
        let zero = if plus || kind == Some("phone") || (pure && core[0].starts_with('0') && core.iter().map(|part| part.chars().count()).sum::<usize>() >= 10) {
            "oh"
        } else {
            "zero"
        };
        let mut said = if pure {
            Self::digit_groups(&core, zero)
        } else {
            core.iter().map(|part| if py::all_digit(part) { digits(part, zero) } else { Self::alnum(part, zero) }).collect::<Vec<_>>().join(" , ")
        };
        if let Some(lead) = lead {
            said = format!("{} , {}", spelled(lead), said);
        }
        if let Some(trail) = trail {
            said = format!("{} , {}", said, spelled(trail));
        }
        format!("{}{}{} {}", head, if plus { " plus " } else { " " }, said, tail)
    }

    /// Letters and digits written together: B12, 14C, X7K, PS5. A single
    /// letter is always named; it is never the article "a".
    fn glued(&self, c: &Captures) -> String {
        let mut out = Vec::new();
        for run in letter_digit_runs(gtext(c, 0)) {
            if py::all_digit(run) {
                out.push(if short_number(run) { cardinal_of(run) } else { digits(run, "zero") });
            } else if run.len() == 1 || ((py::is_upper(run) || run.len() <= 3) && !self.reads_as_word(run)) {
                out.push(spelled(run));
            } else {
                out.push(run.to_string());
            }
        }
        format!(" {} ", out.join(" "))
    }

    // -- web addresses and file names --

    /// One run of letters inside an address: a word, or letters to name.
    fn address_part(&self, part: &str) -> String {
        let lower = part.to_lowercase();
        let length = part.chars().count();
        if length > 1 && self.known(&lower) && (length > 2 || words!("my to in on of go up no do me we us it is at by be so or an as if he hi ok").contains(lower.as_str())) {
            return part.to_string();
        }
        if length <= 3 { spelled(part) } else { part.to_string() }
    }

    fn spoken_address(&self, core: &str) -> String {
        let pieces = rx::find_all(rx!(core, r"[A-Za-z]+|\d+|."), core);
        let mail = core.contains('@');
        let mut host = false;
        let mut out: Vec<String> = Vec::new();
        for (index, piece) in pieces.iter().enumerate() {
            let following = pieces.get(index + 1).copied().unwrap_or("");
            if *piece == "@" {
                host = true;
            }
            if py::all_digit(piece) {
                out.push(address_digits(piece));
            } else if !py::all_alpha(piece) {
                out.push(separator(piece).to_string());
            } else if index > 0
                && pieces[index - 1] == "."
                && (host || !mail)
                && (matches!(following, "" | "/" | ":" | "?" | "#")
                    || (piece.chars().count() <= 3 && (self.n.endings.contains_key(piece.to_lowercase().as_str()) || !self.known(&piece.to_lowercase()))))
            {
                // the ending of a file name or an address: a word if the table says so, otherwise letters
                let lower = piece.to_lowercase();
                out.push(if self.n.endings.get(lower.as_str()) == Some(&"word") { lower } else { spelled(piece) });
            } else {
                let mut letters = String::new();
                for part in rx::find_all(rx!(*piece, r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])"), piece) {
                    let said = self.address_part(part);
                    if said.starts_with(SPELL) {
                        letters.push_str(part);
                        continue;
                    }
                    if !letters.is_empty() {
                        out.push(spelled(&letters));
                        letters.clear();
                    }
                    out.push(said);
                }
                if !letters.is_empty() {
                    out.push(spelled(&letters));
                }
            }
        }
        out.into_iter().filter(|item| !item.is_empty()).collect::<Vec<_>>().join(" ")
    }

    /// Web and mail addresses, file names and paths: the dots, slashes,
    /// underscores and @ are spoken.
    fn address(&self, c: &Captures) -> String {
        let whole = gtext(c, 0);
        let text = whole.trim_end_matches(['.', ',', ';', ':', '!', '?', ')']); // punctuation of the sentence, not of the address
        let tail = &whole[text.len()..];
        if named(c, "file").is_some() {
            let ending = named(c, "ending").unwrap_or("");
            let known = self.n.endings.contains_key(ending.to_lowercase().as_str());
            let marked = named(c, "folder").is_some_and(|f| !f.is_empty()) || named(c, "path").is_some_and(|p| !p.is_empty()) || named(c, "stem").unwrap_or("").contains('_');
            if !(known || (marked && ending.len() <= 5 && (py::is_lower(ending) || py::is_upper(ending)) && !self.known(&ending.to_lowercase()))) {
                return whole.to_string(); // "the end.Then": a missing space, not a file
            }
        }
        let mut core = text;
        for prefix in ["https://", "http://"] {
            if let Some(rest) = core.strip_prefix(prefix) {
                core = rest;
                break;
            }
        }
        core = core.strip_prefix("www.").unwrap_or(core);
        format!(" {} {}", self.spoken_address(core.trim_end_matches('/')), tail)
    }

    /// Chat spellings that contain a digit or a slash: gr8, b4, 2day.
    fn chat_form(&self, c: &Captures) -> String {
        match self.n.chat.get(gtext(c, 0).to_lowercase().as_str()) {
            None => gtext(c, 0).to_string(),
            Some(&"-") => " ".to_string(),
            Some(said) => format!(" {} ", py::split(said).map(|part| part.strip_prefix('*').map_or_else(|| part.to_string(), spelled)).collect::<Vec<_>>().join(" ")),
        }
    }

    /// `{...}`: a pronunciation written by the user.
    fn inline_form(&mut self, c: &Captures) -> String {
        let symbols: Vec<&str> = py::split(gtext(c, 2)).collect();
        if symbols.is_empty() {
            self.error.get_or_insert("empty pronunciation {} in the text".to_string());
            return String::new();
        }
        let phones = if is_arpabet(&symbols) { convert("", &symbols, None) } else { from_text(&symbols.join(" ")) };
        match phones {
            Ok(phones) => {
                self.inline.push((gtext(c, 1).to_string(), phones));
                format!(" {INLINE}{}{INLINE} ", char::from_u32(INLINE_BASE + self.inline.len() as u32 - 1).unwrap_or(INLINE))
            }
            Err(error) => {
                self.error.get_or_insert(error);
                String::new()
            }
        }
    }

    /// `U.S.`, `e.g.`, `p.m.`: letters with full stops.
    fn dotted(text: &str, c: &Captures) -> String {
        let letters = gtext(c, 0).replace('.', "");
        let following = py::lstrip(&text[end(c)..]);
        let said = match letters.to_lowercase().as_str() {
            "eg" => "for example".to_string(),
            "ie" => "that is".to_string(),
            _ => spelled(&letters.to_uppercase()),
        };
        let ends_sentence = sentence_follows(following);
        format!(" {}{} ", said, if ends_sentence && gtext(c, 0).ends_with('.') { " ." } else { "" })
    }

    /// `re.sub(r"\b([A-Za-z]{1,6})\b(\.?)(?![A-Za-z'’])", self._abbreviation, text)`, without the
    /// regular expression: this pass looks at every short word of every text.
    fn abbreviations(&self, text: &str) -> String {
        let word = |c: char| c == '_' || py::is_alnum(c);
        let mut out = String::with_capacity(text.len() + 16);
        let mut copied = 0;
        let mut previous_word = false; // the character before is one that `\w` matches
        let mut chars = text.char_indices().peekable();
        while let Some((from, c)) = chars.next() {
            if !ascii_letter(c) || previous_word {
                previous_word = word(c);
                continue;
            }
            // A run of letters that starts at a word boundary.
            let mut to = from + 1;
            while let Some(&(at, next)) = chars.peek() {
                if !ascii_letter(next) {
                    break;
                }
                to = at + 1;
                chars.next();
            }
            previous_word = true;
            let after = text[to..].chars().next();
            if to - from > 6 || after.is_some_and(word) {
                continue;
            }
            // The full stop belongs to the match unless a letter or an apostrophe follows it.
            let mut stop = after == Some('.');
            if stop && matches!(text[to + 1..].chars().next(), Some(c) if ascii_letter(c) || c == '\'' || c == '’') {
                stop = false;
            }
            if !stop && matches!(after, Some('\'') | Some('’')) {
                continue;
            }
            let end = to + stop as usize;
            if let Some(said) = self.abbreviation(text, from, end, &text[from..to], stop) {
                out.push_str(&text[copied..from]);
                out.push_str(&said);
                copied = end;
            }
            if stop {
                chars.next();
                previous_word = false;
            }
        }
        out.push_str(&text[copied..]);
        out
    }

    /// What an abbreviation is read as, or `None` when the word is not one here.
    fn abbreviation(&self, text: &str, from: usize, to: usize, token: &str, stop: bool) -> Option<String> {
        let key = token.to_lowercase();
        let key = key.as_str();
        if !matches!(key, "st" | "dr") && !self.n.abbreviations.contains_key(key) {
            return None;
        }
        let following = &text[to..];
        // re.match(r"\s*([A-Za-z]+|\d)", following)
        let next_word: Option<&str> = {
            let rest = py::lstrip(following);
            match word_after(rest) {
                Some(word) => Some(word),
                None => py::first(rest).filter(|ch| py::is_decimal(*ch)).map(|ch| &rest[..ch.len_utf8()]),
            }
        };
        let capital_next = next_word.is_some_and(|word| py::first(word).unwrap().is_uppercase());
        let head = &text[..from];
        // re.search(r"[A-Z][a-z]+\s+$", head)
        let capital_before = {
            let body = py::rstrip(head);
            let lower = body.len() - body.trim_end_matches(|ch: char| ch.is_ascii_lowercase()).len();
            body.len() < head.len() && lower > 0 && body[..body.len() - lower].chars().next_back().is_some_and(|ch| ch.is_ascii_uppercase())
        };
        let upper_first = py::first(token).unwrap().is_uppercase();
        if matches!(key, "st" | "dr") && upper_first {
            if capital_next && (stop || key == "dr") {
                return Some(if key == "st" { " saint " } else { " doctor " }.to_string());
            }
            if capital_before && stop {
                return Some(format!("{}{}", if key == "st" { " street" } else { " drive" }, if py::strip(following).is_empty() { " ." } else { " " }));
            }
            return None;
        }
        let Some(&(when, said)) = self.n.abbreviations.get(key) else { return None };
        let mut said = said;
        let number_next = next_word.is_some_and(py::all_digit);
        if when == "title" {
            if !(upper_first && (capital_next || stop)) {
                return None;
            }
        } else if when == "number" {
            // re.search(r"([A-Za-z]+)\.?\s+$", head)
            let previous = {
                let body = py::rstrip(head);
                if body.len() < head.len() {
                    let body = body.strip_suffix('.').unwrap_or(body);
                    let length = body.len() - body.trim_end_matches(ascii_letter).len();
                    if length > 0 { Some(&body[body.len() - length..]) } else { None }
                } else {
                    None
                }
            };
            let numbered = words!("ref reference serial account order invoice tracking part model phone tel fax id room flight case \
                                   ticket policy licence license booking customer vat tax reg registration");
            if !(stop && (number_next || (key == "no" && previous.is_some_and(|word| numbered.contains(word.to_lowercase().as_str()))))) {
                return None;
            }
        } else if !stop || (when == "small" && py::is_upper(token)) {
            return None;
        } else if matches!(key, "mar" | "sun" | "sat" | "jan" | "dec" | "max" | "min" | "est" | "no") && !upper_first && !number_next {
            // also words; "max. load" and "est. time" are the abbreviation, "he sat." is not
            if !(matches!(key, "max" | "min" | "est") && next_word.is_some_and(|word| py::first(word).unwrap().is_lowercase())) {
                return None;
            }
        }
        if key == "est" && !number_next {
            said = "estimated"; // "Est. arrival"; "est. 1885" is "established"
        }
        let ends_sentence = when != "title"
            && (py::strip(following).is_empty() || (capital_next && next_word.is_some_and(|word| words!(SENTENCE_STARTERS).contains(word.to_lowercase().as_str()))));
        Some(format!(" {}{} ", said, if ends_sentence { " ." } else { "" }))
    }

    /// The text with every non-word written out, markers included.
    fn rewrite(&mut self, text: &str) -> String {
        // before "1½" becomes "11/2" and "m²" becomes "m2"
        // Each pass is skipped when the text lacks a character every match of it needs.
        let mut text = if text.is_ascii() {
            text.to_string()
        } else {
            sub(rx!(text, r"(?<![\d.,/])(?:(\d+)\s?)?([½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅐⅛⅜⅝⅞⅑⅒])"), text, Self::vulgar)
        };
        if text.contains(['²', '³']) {
            text = sub(rx!(&text, r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft|mi|yd|in)([²³])"), &text, |c| self.square(c));
        }
        text = text.replace('²', " squared ").replace('³', " cubed ");
        text = text.nfkc().collect::<String>().replace('\u{2044}', "/").replace('\u{2212}', " minus ");
        if !text.is_ascii() {
            let pattern = rx!(&text, r"(?i)\b(résumé|resumé|exposé|rosé|pâté|maté|saké|lamé)(s?)\b");
            text = sub(pattern, &text, |c| {
                format!("{{{}{}}}", accent_decides(&gtext(c, 1).to_lowercase()).unwrap_or(""), if gtext(c, 2).is_empty() { "" } else { " Z" })
            });
            let mut plain = String::with_capacity(text.len());
            for ch in text.chars() {
                plain_letter(ch, &mut plain);
            }
            text = plain;
        }
        if text.contains('–') {
            text = sub(rx!(&text, r"(?<=\d)–(?=\d)"), &text, |_| "-".to_string());
        } // 10–15 is a range, not a pause
        if text.contains(['-', '–']) {
            let days = rx!(&text, format!(r"\b({DAY_PATTERN}|{MONTH_PATTERN})\b(\.?)\s?[-–]\s?({DAY_PATTERN}|{MONTH_PATTERN})\b(\.?)"));
            text = sub(days, &text, |c| self.day_range(&text, c));
        }
        if !text.is_ascii() || text.contains(['`', '[', ']']) {
            text = text
                .chars()
                .map(|ch| match ch {
                    '‘' | '’' | '‚' | '‛' | '`' | '´' => '\'',
                    '“' | '”' | '„' | '‟' | '«' | '»' => '"',
                    '–' | '―' => '—',
                    '[' => '(',
                    ']' => ')',
                    other => other,
                })
                .collect();
        }
        self.inline.clear();
        if text.contains('{') {
            text = sub(rx!(&text, r"([A-Za-z']+)?\{([^{}]*)\}"), &text, |c| self.inline_form(c));
        }
        text = text.replace('{', "(").replace('}', ")");
        if may_hold_address(&text) {
            let addresses = rx!(
                &text,
                concat!(
                    r#"(?P<url>(?:https?://|www\.)[^\s<>"']+)"#,
                    r"|(?P<mail>[\w.+-]+@[\w-]+(?:\.[\w-]+)+)",
                    r"|(?P<file>(?<![\w@./-])(?P<folder>/?(?:[\w~-]+/)+)?(?P<stem>[\w-]+(?:\.[\w-]+)*)\.(?P<ending>[A-Za-z][A-Za-z0-9]{0,9})",
                    r#"(?![\w-])(?P<path>/[^\s<>"']*)?)"#
                )
            );
            text = sub(addresses, &text, |c| self.address(c));
        }
        text = text.replace('→', " to ");
        if !text.is_ascii() {
            // Emoji: runs of pictographs, arrows, variation selectors and joiners.
            let emoji = |ch: char| {
                matches!(ch as u32, 0x1F000..=0x1FAFF | 0x2600..=0x27BF | 0x2190..=0x21FF | 0x2B00..=0x2BFF | 0xFE0F | 0x200D | 0x20E3)
            };
            let mut out = String::with_capacity(text.len());
            let mut inside = false;
            for ch in text.chars() {
                if emoji(ch) {
                    if !inside {
                        out.push(' ');
                    }
                    inside = true;
                } else {
                    inside = false;
                    out.push(ch);
                }
            }
            text = out;
        }
        if text.contains([':', ';', '=', '<', '^', '_', '\\']) {
            let emoticons = rx!(
                &text,
                concat!(
                    r"(?<![A-Za-z0-9])(?:[:;=]['^-]?[DPpOo3|/\\*\]\[}{><]+|[)(\]\[][-^']?[:;=]|<3+|</3|\^_?\^|-_-|[oO]_[oO]|>_<|T_T|\\o/|:'[()])(?![A-Za-z0-9])",
                    r"|[:;=]['^-]?[)(]+"
                )
            );
            text = sub(emoticons, &text, |_| " ".to_string());
        }
        if text.contains("...") {
            text = sub(rx!(&text, r"\.{3,}"), &text, |_| "…".to_string());
        }
        if text.contains('-') {
            text = sub(rx!(&text, r"(?<=\d)\s-\s(?=\d[\d.,]*\s?=)"), &text, |_| " minus ".to_string());
            text = sub(rx!(&text, r"-{2,}|\s-\s"), &text, |_| "—".to_string());
        }
        if text.contains('/') {
            text = sub(rx!(&text, r"(?i)\bw/o\b"), &text, |_| " without ".to_string());
            text = sub(rx!(&text, r"(?i)\bw/(?=\s|$)"), &text, |_| " with ".to_string());
            text = sub(rx!(&text, r"(?i)\bn/a\b"), &text, |_| " not applicable ".to_string());
        }
        text = text.replace("'n'", " 'n ");
        if text.chars().any(py::is_decimal) {
            // most text has no digit and skips these passes
            text = self.figures(text);
        }
        self.signs(text)
    }

    /// The passes that need a digit: money, times, dates, telephone numbers,
    /// versions, codes, units, fractions, ranges.
    fn figures(&mut self, mut text: String) -> String {
        if text.contains(['$', '£', '€', '¥', '₹']) || has_currency_code(&text) {
            let money = rx!(
                &text,
                format!(
                    r"(?:(?<![\w.])(?P<minus>-)\s?)?(?P<symbol>[$£€¥₹]|(?<![A-Za-z])(?:{CURRENCY_CODES})(?![A-Za-z]))\s?(?P<inner>-)?{AMOUNT}(?:\s?(?P<scale>thousand|million|billion|trillion|bn|[kKmMbB])(?![A-Za-z]))?"
                )
            );
            text = sub(money, &text, |c| self.money(c));
        }
        if has_currency_code(&text) {
            text = sub(rx!(&text, format!(r"(?<![\w.,]){AMOUNT}\s?(?P<symbol>{CURRENCY_CODES})(?![A-Za-z])")), &text, |c| self.money(c));
        }
        if text.contains('¢') {
            text = sub(rx!(&text, r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?¢"), &text, |c| {
                format!(" {} {} ", number_words(gtext(c, 1)), if gtext(c, 1) == "1" { "cent" } else { "cents" })
            });
        }

        // times of day, lengths of time, ratios
        if text.contains('T') {
            text = sub(rx!(&text, format!(r"({CLOCK})\s({ZONES})\b")), &text, |c| format!("{} {}", gtext(c, 1), spelled(gtext(c, 2))));
        }
        if has_decimal_point(&text) {
            text = sub(rx!(&text, r"(?<![\d.,:])(\d{1,2})\.([0-5]\d)(?=\s?[AaPp]\.?[Mm]\b)"), &text, |c| format!("{}:{}", gtext(c, 1), gtext(c, 2))); // 7.30 pm
            text = sub(
                rx!(&text, r"(?<![\d.,:])(\d{1,2})\.(00|15|30|45)(\s?(?:to|until|till|and|[-—])\s?)(\d{1,2})\.(00|15|30|45)(?!\d|\.\d)"),
                &text,
                |c| self.dot_times(&text, c),
            );
        }
        if text.contains(['-', '—']) {
            text = sub(rx!(&text, format!(r"({CLOCK})\s?[-—]\s?(?=\d{{1,2}}(?::[0-5]\d|\s?[AaPp]\.?[Mm]\b))")), &text, |c| format!("{} to ", gtext(c, 1)));
        }
        if text.contains(':') {
            text = sub(rx!(&text, r"(?<![\d:.,])(\d{1,3}):(\d{1,3}(?:,\d{3})+|\d+)(?![\d:]|[.,]\d)"), &text, |c| self.ratio(&text, c));
            text = sub(rx!(&text, format!(r"(?<![\d:])(\d{{1,2}}):([0-5]\d)(?::([0-5]\d))?{HALF_DAY}?(?![\d:])")), &text, |c| self.time(&text, c));
        }
        if text.contains(['m', 'M']) {
            text = sub(rx!(&text, format!(r"(?<![\d.:,])(\d{{1,2}}){HALF_DAY}(?![A-Za-z])")), &text, |c| {
                format!(" {} {}", cardinal(py::int(gtext(c, 1)) as u128), Self::half_day(&text, c, group(c, 2), group(c, 3)))
            });
        }

        // dates
        let mut seen: Vec<(u64, u64)> = Vec::new();
        let slashes = text.contains('/');
        if slashes {
            for c in rx!(&text, r"(?<![\d/.])(\d{1,2})/(\d{1,2})(?:/(?:\d{4}|\d{2}))?(?![\d/])").captures_iter(&text).flatten() {
                let pair = (py::int(gtext(&c, 1)), py::int(gtext(&c, 2)));
                if pair != (24, 7) {
                    seen.push(pair);
                }
            }
        }
        self.day_first = seen.iter().any(|&(a, b)| a > 12 && 12 >= b) && !seen.iter().any(|&(a, b)| b > 12 && 12 >= a);
        self.dated = slashes && rx::found(rx!(&text, r"(?<![\d/])\d{1,2}/\d{1,2}/(?:\d{4}|\d{2})(?![\d/])"), &text);
        let year_first = |c: &Captures| Self::date(py::int(gtext(c, 2)), py::int(gtext(c, 3)), group(c, 1), false).unwrap_or_else(|| gtext(c, 0).to_string());
        if text.contains('-') {
            text = sub(rx!(&text, r"(?<![\d-])(\d{4})-(\d{1,2})-(\d{1,2})(?![\d-])"), &text, year_first);
        }
        if text.contains('/') {
            text = sub(rx!(&text, r"(?<![\d/])(\d{4})/(\d{1,2})/(\d{1,2})(?![\d/])"), &text, year_first);
        }
        if has_decimal_point(&text) {
            text = sub(rx!(&text, r"(?<![\d.])(\d{1,2})\.(\d{1,2})\.(\d{4})(?!\d|\.\d)"), &text, Self::dot_date);
        }
        if text.contains('/') {
            text = sub(rx!(&text, r"(?<![\d/])(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})(?![\d/])"), &text, |c| self.slash_date(c));
        }
        if has_month(&text) {
            text = sub(
                rx!(&text, format!(r"\b({MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?\s?[-—]\s?(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?(?![\d:])")),
                &text,
                Self::days_of_month,
            );
            text = sub(
                rx!(&text, format!(r"(?<![\d.,/:-])(\d{{1,2}})(?:st|nd|rd|th)?\s?[-—]\s?(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})\b(?:,?\s+(\d{{4}}))?")),
                &text,
                Self::days_of_month,
            );
            text = sub(rx!(&text, format!(r"\b({MONTH_PATTERN})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?(?![\d:])")), &text, |c| {
                Self::date(month_number(gtext(c, 1)).map_or(0, |m| m as u64 + 1), py::int(gtext(c, 2)), group(c, 3), false).unwrap_or_else(|| gtext(c, 0).to_string())
            });
            text = sub(
                rx!(&text, format!(r"(?<![\d.,/:-])(\d{{1,2}})(?:st|nd|rd|th)?\s+({MONTH_PATTERN})\b(\.(?=\s+[a-z\d]))?(?:,?\s+(\d{{4}}))?")),
                &text,
                Self::day_month,
            );
        }

        // telephone numbers, versions, codes
        text = sub(rx!(&text, r"(?<![\d-])(?:\+?(1)[-.\s])?\(?(\d{3})\)?[-.\s](\d{3})[-.\s](\d{4})(?![\d-])"), &text, Self::phone);
        if text.contains('-') {
            text = sub(rx!(&text, r"(?<![\w.,-])(\d{3})-(\d{4})(?![\w-])"), &text, Self::phone);
        }
        if has_decimal_point(&text) {
            text = sub(rx!(&text, r"(?<![\w.])([vV])?(\d+(?:\.\d+){2,})(?:-(\d+)(?![\w-]|\.\d))?(?![\w])"), &text, |c| {
                format!(
                    " {}{}{} ",
                    group(c, 1).map(|v| format!("{} ", spelled(v))).unwrap_or_default(),
                    gtext(c, 2).split('.').map(version_part).collect::<Vec<_>>().join(" point "),
                    group(c, 3).filter(|g| !g.is_empty()).map(|g| format!(" dash {}", number_words(g))).unwrap_or_default()
                )
            });
            text = sub(rx!(&text, r"(?<![\w.])([vV])(\d+)\.(\d{2,3})(?!\d|\.\d|\w)"), &text, |c| {
                format!(" {} {} point {} ", spelled(gtext(c, 1)), cardinal_of(gtext(c, 2)), minor(gtext(c, 3)))
            });
            // The original lists the names in the pattern, without regard to case:
            //   (?i)\b(version|ver|...)(\.?\s(?:(?:from|to)\s)?)(\d+)\.(\d{2,3})(?:(\s(?:to|or|and)\s)(\d+)\.(\d{2,3}))?(?!...)
            // A name there is always a whole run of letters, so the run is matched here and
            // looked up; that pattern is slow to build, and this one gives the same matches.
            text = sub(
                rx!(
                    &text,
                    concat!(
                        r"\b([A-Za-z]+)(\.?\s(?:(?:[Ff][Rr][Oo][Mm]|[Tt][Oo])\s)?)(\d+)\.(\d{2,3})",
                        r"(?:(\s(?:[Tt][Oo]|[Oo][Rr]|[Aa][Nn][Dd])\s)(\d+)\.(\d{2,3}))?",
                        r"(?!\d|\.\d|\s?%|\s(?:[Pp][Ee][Rr][Cc][Ee][Nn][Tt]|[Mm][Ii][Ll][Ll][Ii][Oo][Nn]|[Bb][Ii][Ll][Ll][Ii][Oo][Nn])\b)"
                    )
                ),
                &text,
                |c| {
                    if !versioned(gtext(c, 1)) {
                        return gtext(c, 0).to_string();
                    }
                    let second = match group(c, 5).filter(|g| !g.is_empty()) {
                        Some(link) => format!("{}{} point {}", link, cardinal_of(gtext(c, 6)), minor(gtext(c, 7))),
                        None => String::new(),
                    };
                    format!("{}{}{} point {}{} ", gtext(c, 1), gtext(c, 2), cardinal_of(gtext(c, 3)), minor(gtext(c, 4)), second)
                },
            );
        }
        if text.contains('-') {
            text = sub(rx!(&text, r"(?<![\w.,/-])(#)?([A-Za-z0-9]+(?:-[A-Za-z0-9]+)+)(?![\w-]|[.,]\d)"), &text, |c| self.code(&text, c));
        }
        if text.contains('/') {
            text = sub(rx!(&text, r"(?<![\w.,/-])(\d+(?:/\d+){2,})(?![\w/]|[.,]\d)"), &text, |c| {
                format!(" {} ", Self::digit_groups(&gtext(c, 1).split('/').collect::<Vec<_>>(), "zero"))
            });
        }
        // The conditional group of the original, (?(lead)...), written as two alternatives.
        text = sub(
            rx!(
                &text,
                concat!(
                    r"(?<![\w.,:/#+-])(?P<plus>\+)?",
                    r"(?:(?P<lead>[A-Z]{2}) (?P<core1>[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+)(?: (?P<trail>[A-Z]{2})(?![\w]))?",
                    r"|(?P<core2>[A-Z0-9]*\d[A-Z0-9]*(?: [A-Z0-9]*\d[A-Z0-9]*)+))",
                    r"(?![\w]|[.,:/-]\d|\s?[%°])"
                )
            ),
            &text,
            |c| self.grouped(&text, c),
        );

        if text.contains('-') {
            text = sub(rx!(&text, r"(?<![\w.])-(?=\d)"), &text, |_| " minus ".to_string());
        }
        if text.contains(['x', 'X', '×']) {
            text = sub(rx!(&text, format!(r"(?:(?<=\d)\s?[xX×]|×)\s?(?=\d[\d.,]*\s?(?:{LENGTHS})\b)")), &text, |_| " by ".to_string()); // 3 × 4 metres
            text = sub(rx!(&text, r"(?<=\d)\s[xX×]\s(?=\d)"), &text, |_| " times ".to_string());
        }
        text = sub(rx!(&text, r"(?<![\w/'])[A-Za-z0-9/]*[0-9/][A-Za-z0-9/]*(?![\w/'])"), &text, |c| self.chat_form(c));
        if text.contains('s') {
            text = sub(rx!(&text, r"(?<![\w.])'?(\d{2}|\d{4})'?s\b"), &text, |c| {
                let number = py::int(gtext(c, 1)) as u128;
                format!(" {} ", plural_of(&if gtext(c, 1).chars().count() == 4 { year(number) } else { cardinal(number) }))
            });
        }
        if text.contains(['t', 'd']) {
            text = sub(rx!(&text, r"(?<![\w.])(\d+)(?:st|nd|rd|th)\b"), &text, |c| format!(" {} ", ordinal_of(gtext(c, 1))));
        }
        if text.contains(['p', 'i']) {
            text = sub(rx!(&text, r"(?<![\w.])(144|240|360|480|540|576|720|900|1080|1440|2160|4320)([pi])\b"), &text, |c| {
                format!(" {} {} ", pairs(py::int(gtext(c, 1)) as u128), spelled(gtext(c, 2))) // 1080p
            });
        }
        if text.contains('%') {
            text = sub(rx!(&text, r"(\d[\d,]*(?:\.\d+)?)\s?%"), &text, |c| format!(" {} percent ", number_words(gtext(c, 1))));
        }
        text = sub(rx!(&text, r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)\s?(km|cm|mm|m|ft)([23])(?![\w])"), &text, |c| self.square(c));
        let units = rx!(&text, {
            let number = r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?:[-–](\d[\d,]*(?:\.\d+)?))?\s?";
            if rx::is_wide(&text) {
                format!(r"(?i){number}({})(?![A-Za-z])(\.(?=\s+(?-i:[a-z])))?", self.n.unit_names)
            } else {
                format!(r"{number}({})(?![A-Za-z])(\.(?=\s+[a-z]))?", self.n.unit_names_ascii)
            }
        });
        text = sub(units, &text, |c| self.measure(&text, c));
        if text.contains('/') {
            text = sub(rx!(&text, r"(?<![\d/.,-])(\d+) (\d)/(2|3|4|5|6|8|10|16)(?![\d/]|\.\d)"), &text, Self::mixed_fraction);
            text = sub(rx!(&text, r"(?<![\d/.])(\d+)/(\d+)(?![\d/]|\.\d)"), &text, |c| self.fraction(&text, c));
        }
        if text.contains(['-', '–']) {
            text = sub(rx!(&text, r"(?<![\w.,-])(\d{4})[-–](\d{4})(?![\w.,-])"), &text, |c| {
                format!(" {} to {} ", year(py::int(gtext(c, 1)) as u128), year(py::int(gtext(c, 2)) as u128))
            });
            text = sub(rx!(&text, r"(?<![\w.,-])(\d+)[-–](\d+)(?![\w-]|[.,]\d)"), &text, |c| {
                format!(" {} to {} ", number_words(gtext(c, 1)), number_words(gtext(c, 2)))
            });
        }
        if text.contains('x') {
            text = sub(rx!(&text, r"(?<![\w.])(\d+)x\b"), &text, |c| format!(" {} times ", number_words(gtext(c, 1))));
        }
        if text.contains('^') {
            text = sub(rx!(&text, r"(?<=\d)\^(?=\d)"), &text, |_| " to the power of ".to_string());
        }
        text
    }

    /// The passes that follow: signs, abbreviations, and the numbers that are left.
    fn signs(&mut self, mut text: String) -> String {
        if text.contains('-') {
            text = sub(rx!(&text, format!(r"(?<![\w-])({SIZES})-({SIZES})(?![\w-])")), &text, |c| {
                format!(" {} to {} ", spelled(gtext(c, 1)), spelled(gtext(c, 2)))
            });
        }
        if text.contains('#') {
            text = sub(rx!(&text, r"#(?=\d)"), &text, |_| " number ".to_string());
            text = sub(rx!(&text, r"#(?=[A-Z]{1,4}-?\d)"), &text, |_| " number ".to_string()); // Order #A7
            text = sub(rx!(&text, r"#(?=[A-Za-z])"), &text, |_| " hashtag ".to_string());
        }
        if text.contains('~') {
            text = sub(rx!(&text, r"~(?=\s?\d)"), &text, |_| " about ".to_string());
        }
        if text.contains('=') {
            text = sub(rx!(&text, r"(?<=[\w\s])(>=|<=|!=)(?=[\w\s])"), &text, |c| {
                format!(" {} ", match gtext(c, 1) { ">=" => "greater than or equal to", "<=" => "less than or equal to", _ => "not equal to" })
            });
        }
        if text.contains(" Ti") {
            text = sub(rx!(&text, r"(?<=\d) Ti\b"), &text, |_| format!(" {}", spelled("TI")));
        } // RTX 4070 Ti
        if text.contains('.') {
            text = sub(rx!(&text, r"\b([Rr])ef\.\s?[Nn]o\.(?=\s)"), &text, |c| format!("{}eference number", gtext(c, 1)));
            text = sub(rx!(&text, r"\b((?:[A-Za-z]\.){2,})"), &text, |c| Self::dotted(&text, c));
        }
        text = self.abbreviations(&text);
        if text.chars().any(py::is_decimal) {
            text = sub(rx!(&text, r"(?<![\w'.])(?=[A-Za-z0-9]*[A-Za-z])(?=[A-Za-z0-9]*\d)[A-Za-z0-9]+(?![\w']|\.\d)"), &text, |c| self.glued(c));
            text = sub(rx!(&text, r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|(?<![\w.])\.\d+"), &text, |c| self.number(&text, c));
        }
        if text.contains('/') {
            text = sub(rx!(&text, r"(?<=[A-Za-z])/(?=[A-Za-z])"), &text, |_| " ".to_string());
        }
        if text.contains(['<', '>']) {
            text = sub(rx!(&text, r"(?<=\d)\s?[<>]\s?(?=\d)|(?<=\s)[<>](?=\s)"), &text, |c| {
                if gtext(c, 0).contains('<') { " less than " } else { " greater than " }.to_string()
            });
        }
        if text.chars().any(|ch| symbol_words(ch).is_some()) {
            let mut out = String::with_capacity(text.len() + 16);
            for ch in text.chars() {
                match symbol_words(ch) {
                    Some(said) => {
                        out.push(' ');
                        out.push_str(said);
                        out.push(' ');
                    }
                    None => out.push(ch),
                }
            }
            text = out;
        }
        text
    }

    // -- words --

    /// Spellings a word stretched for emphasis may stand for. A letter typed
    /// three times or more is cut to one or to two.
    fn unstretched(word: &str) -> Vec<String> {
        let chars: Vec<char> = word.chars().collect();
        // Runs of one character: (character, length).
        let mut runs: Vec<(char, usize)> = Vec::new();
        for &ch in &chars {
            match runs.last_mut() {
                Some((last, count)) if *last == ch => *count += 1,
                _ => runs.push((ch, 1)),
            }
        }
        if !runs.iter().any(|&(ch, count)| count >= 3 && ch != '\n') {
            return Vec::new();
        }
        let cut = |double_inside: bool, double_final: bool| -> String {
            let mut out = String::new();
            for (index, &(ch, count)) in runs.iter().enumerate() {
                if count >= 3 && ch != '\n' {
                    let last = index == runs.len() - 1;
                    let double = if last { double_final } else { double_inside };
                    let keep = if double { 2 } else { 1 };
                    out.extend(std::iter::repeat(ch).take(keep));
                } else {
                    out.extend(std::iter::repeat(ch).take(count));
                }
            }
            out
        };
        let (last, count) = *runs.last().unwrap();
        let collapsed: String = {
            let mut out: String = runs[..runs.len() - 1].iter().flat_map(|&(ch, n)| std::iter::repeat(ch).take(n)).collect();
            out.extend(std::iter::repeat(last).take(if count >= 3 { 1 } else { count }));
            out
        };
        let liked = (count >= 3 && matches!(last, 'l' | 'f' | 's' | 'z')) || collapsed == "to";
        let mut ordered: Vec<String> = Vec::new();
        for candidate in [cut(true, liked), cut(true, !liked), cut(false, liked), cut(false, !liked)] {
            if !ordered.contains(&candidate) {
                ordered.push(candidate);
            }
        }
        ordered
    }

    /// Tokens for one run of letters and apostrophes as written. `previous`
    /// is the word before it, `following` whether a word follows directly,
    /// `fresh` whether a new sentence starts here.
    fn word_tokens(&self, raw: &str, all_caps: bool, previous: &str, following: bool, fresh: bool) -> Vec<Word> {
        let lower = raw.to_lowercase();
        if let Some(&chat) = self.n.chat.get(lower.as_str()) {
            if raw.len() > 1 || py::is_lower(raw) {
                if chat == "-" {
                    return Vec::new();
                }
                return py::split(chat)
                    .map(|part| match part.strip_prefix('*') {
                        Some(letters) => Word::spelled(letters.to_lowercase(), raw),
                        None => Word::new(part.to_lowercase(), part),
                    })
                    .collect();
            }
        }
        let previous_capital = py::first(previous).is_some_and(|ch| ch.is_uppercase());
        if let Some(number) = roman(raw) {
            if previous_capital {
                let said = if words!("war chapter part book volume act scene section phase type class level stage title article appendix world")
                    .contains(previous.to_lowercase().as_str())
                {
                    cardinal(number)
                } else {
                    format!("the {}", ordinal(number))
                };
                return said.split(' ').map(|word| Word::new(word, raw)).collect();
            }
        }
        let letters = raw.replace('\'', "");
        let possessive = lower.ends_with("'s");
        let stem = if possessive { &raw[..raw.len() - 2] } else { raw };
        if stem.len() == 1 && py::all_alpha(stem) {
            // "A" is the article unless it labels something: "Plan A.", "vitamin A"
            let label = stem == "A"
                && !previous.is_empty()
                && (!following
                    || (!fresh
                        && words!("plan vitamin grade type option class section part team group exhibit appendix point model series hepatitis level")
                            .contains(previous.to_lowercase().as_str())));
            if matches!(stem, "a" | "A" | "I" | "i") && !possessive && !label {
                return vec![Word::new(stem.to_lowercase(), raw)];
            }
            return vec![Word::spelled(lower, raw)];
        }
        let stem_lower = stem.to_lowercase();
        if py::is_upper(stem) && py::all_alpha(stem) && (2..=5).contains(&stem.len()) && (!all_caps || !self.known(&stem_lower)) {
            if words!("US IT AI ID UN EU UK LA AD BC AM PM OS PE PA MA LED SAT RIP ER").contains(stem) || !self.known(&stem_lower) {
                return vec![Word::spelled(lower, raw)];
            }
        }
        if !py::all_alpha(&letters) {
            return Vec::new();
        }
        // iOS, macOS, eSIM: small letters, then capitals that are named.
        let capitals = raw.bytes().any(|b| b.is_ascii_uppercase()) && rx::found(rx!(raw, r"^[a-z]+(?:[A-Z][a-z]+)*[A-Z]{2,}(?:'s)?\z"), raw);
        if capitals || !self.known(&lower) {
            if !capitals {
                for candidate in Self::unstretched(&lower) {
                    // "soooo" -> "so", "goooood" -> "good"
                    if self.known(&candidate) {
                        return vec![Word::new(candidate, raw)];
                    }
                }
            }
            let parts = rx::find_all(rx!(raw, r"[A-Z]?[a-z']+|[A-Z]+(?![a-z])"), raw);
            if parts.len() > 1 && parts.concat() == raw {
                // iPhone, YouTube, JavaScript
                let mut out: Vec<Word> = Vec::new();
                for part in parts {
                    let named = capitals && py::is_upper(part) && words!("US IT AI ID UN EU UK LA AD BC AM PM OS PE PA MA LED SAT RIP ER").contains(part);
                    let spell = part.len() == 1 || named || (py::is_upper(part) && !self.known(&part.to_lowercase()));
                    match out.last_mut() {
                        Some(last) if capitals && last.spell && (spell || part == "'s") => {
                            // i + OS
                            last.text.push_str(&part.to_lowercase());
                            last.raw.push_str(part);
                        }
                        _ => {
                            let attach = !out.is_empty();
                            out.push(Word { text: part.to_lowercase(), attach, raw: part.to_string(), spell, phones: None });
                        }
                    }
                }
                return out;
            }
        }
        vec![Word::new(lower, raw)]
    }

    fn tokens(&mut self, source: &str) -> Result<Vec<Token>, String> {
        let text = self.rewrite(source);
        if let Some(error) = self.error.take() {
            return Err(error);
        }
        let text = text.as_str();
        // re.findall(r"[A-Za-z]{2,}", text)
        let runs: Vec<&str> = letter_runs(text).into_iter().filter(|run| run.len() >= 2).collect();
        let all_caps = runs.len() >= 3 && runs.iter().filter(|run| py::is_upper(run)).count() as f64 >= 0.6 * runs.len() as f64;
        let mut out: Vec<Token> = Vec::new();
        let pattern = rx!(text, r"\x{E000}([A-Za-z ]+)\x{E001}|\x{E003}(.)\x{E003}|('?[A-Za-z]+(?:'[A-Za-z]+)*)|(\d+)|(-)|(\S)");
        let mut attach = false;
        let mut previous = String::new();
        let mut fresh = true; // a sentence starts here: "a plan. A dog" is the article
        for c in pattern.captures_iter(text) {
            let c = c.map_err(|error| error.to_string())?;
            let (from, to) = (start(&c), end(&c));
            let before_char = text[..from].chars().next_back().unwrap_or(' ');
            let after_char = text[to..].chars().next();
            if group(&c, 5).is_some() {
                attach = matches!(out.last(), Some(Token::Word(_))) && !py::is_space(before_char) && after_char.is_some_and(py::is_alnum);
                continue;
            }
            let mut new: Vec<Word> = Vec::new();
            if let Some(letters) = group(&c, 1) {
                new.push(Word::spelled(letters.replace(' ', "").to_lowercase(), letters));
            } else if let Some(inline) = group(&c, 2) {
                let index = (py::first(inline).unwrap() as u32).wrapping_sub(INLINE_BASE) as usize;
                let (written, phones) = self.inline.get(index).ok_or("stray private-use character in the text")?;
                new.push(Word { phones: Some(phones.clone()), ..Word::new(written.to_lowercase(), written.as_str()) });
            } else if let Some(mut word) = group(&c, 3) {
                if word.starts_with('\'') && !matches!(word.to_lowercase().as_str(), "'em" | "'tis" | "'twas" | "'til" | "'cause" | "'bout" | "'round" | "'n") {
                    out.push(Token::Punct(punctuation("'").unwrap()));
                    word = &word[1..];
                } else if word.starts_with('\'') && !self.known(&word.to_lowercase()) {
                    word = &word[1..];
                }
                let follows = py::lstrip(&text[to..]).chars().next().is_some_and(ascii_letter);
                new = self.word_tokens(word, all_caps, &previous, follows, fresh);
                previous = word.to_string();
            } else if let Some(number) = group(&c, 4) {
                new = number_words(number).split(' ').filter(|part| !part.is_empty()).map(|part| Word::new(part, number)).collect();
            } else if let Some(mark) = group(&c, 6).and_then(punctuation) {
                if !(mark == "'" && py::is_alnum(before_char) && after_char.is_some_and(py::is_alnum)) {
                    out.push(Token::Punct(mark));
                }
                if matches!(mark, "." | "!" | "?" | ";" | "…") {
                    fresh = true;
                }
                attach = false;
                continue;
            }
            if attach {
                if let Some(first) = new.first_mut() {
                    first.attach = true;
                }
            }
            fresh = fresh && new.is_empty();
            out.extend(new.into_iter().map(Token::Word));
            attach = false;
        }
        Ok(out)
    }
}
