"""Check that the Rust port of dizione gives the tokens of the Python original.

    .venv/bin/python scripts/dizione_parity.py [--fixture] [--time] [--show N]

Builds a corpus (the TEXT lines of dizione's tricky_sentences.txt, the string
literals of its tests, scripts/dizione_corpus_extra.txt and generated
sentences), phonemizes every line with both implementations and compares ids
and stress exactly; a text Python rejects must be rejected by Rust.

--fuzz N   add N texts made by shuffling the pieces of the corpus together (--seed S)
--fixture  also write crates/dizione/tests/fixture.jsonl (expected output for `cargo test`)
--time     also compare speed
"""
import argparse, ast, json, os, random, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "weights" / "phonemizer"
os.environ.setdefault("DIZIONE_DATA", str(DATA))
UPSTREAM = ROOT / "upstream" / "dizione"


def corpus():
    out = []
    for line in (UPSTREAM / "tests/data/tricky_sentences.txt").read_text(encoding="utf-8").splitlines():
        if line.startswith("TEXT:"):
            out.append(line[5:].strip())
    for node in ast.walk(ast.parse((UPSTREAM / "tests/test_dizione.py").read_text(encoding="utf-8"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.strip() and len(node.value) < 400:
            out.append(node.value)
    out += [line for line in (ROOT / "scripts/dizione_corpus_extra.txt").read_text(encoding="utf-8").split("\n")]
    out += generated()
    seen, unique = set(), []
    for text in out:
        if text not in seen:
            seen.add(text)
            unique.append(text)
    return unique


def generated():
    r = random.Random(20261115)
    out = []
    onsets = ["", "b", "bl", "br", "ch", "d", "dr", "f", "fl", "fr", "g", "gl", "gr", "h", "j", "k", "kl", "kr", "l", "m", "n", "p", "ph",
              "pl", "pr", "qu", "r", "s", "sc", "sh", "sk", "sl", "sm", "sn", "sp", "st", "str", "sw", "t", "th", "tr", "v", "w", "wh", "x",
              "y", "z", "zh", "kn", "ps", "wr", "sch", "thr", "spl"]
    nuclei = ["a", "e", "i", "o", "u", "y", "ai", "au", "ea", "ee", "ei", "ie", "oa", "oo", "ou", "ue", "ay", "oy", "ew", "ow", "eau", "igh"]
    codas = ["", "b", "ck", "d", "f", "g", "k", "l", "ll", "m", "n", "ng", "nk", "nt", "p", "r", "rd", "rk", "rm", "rn", "rt", "s", "sh",
             "ss", "st", "t", "th", "tch", "x", "z", "ght", "mb", "lm", "que", "gue", "tion", "ous", "ture", "ism", "ology"]
    def word():
        return "".join(r.choice(onsets) + r.choice(nuclei) + r.choice(codas) for _ in range(r.randint(1, 3)))
    frames = ["The {} is a {}.", "I saw {} and {} yesterday.", "{} {} {}!", "Mr. {} met Dr. {} at the {}.", "{}'s {} was {}ed and {}ing.",
              "{}s, {}ly, un{}, re{}, {}ness.", "{}-{} and {}.", "Is {} near {}?"]
    for _ in range(260):
        frame = r.choice(frames)
        words = [word() for _ in range(frame.count("{}"))]
        if r.random() < 0.3:
            words = [w.capitalize() for w in words]
        if r.random() < 0.08:
            words = [w.upper() for w in words]
        out.append(frame.format(*words))
    names = ("Aaliyah Bartholomew Cthulhu Daenerys Eyjafjallajokull Fitzgerald Guillermo Hermione Ignatius Joaquin Kowalczyk Llewellyn "
             "Mbappe Nkemelu Oyelowo Ptolemy Quetzalcoatl Rasmussen Saoirse Thibault Ulyanov Vercingetorix Wojciech Xiaoping Yevgeny "
             "Zbigniew Tsvetkova Lindqvist Papadopoulos Villanueva Fotheringham Bhattacharya Oyinlola Jorgensen Dvorak Szczesny").split()
    for name in names:
        out.append(f"{name} said hello to {r.choice(names)}'s friend.")
    def number():
        kind = r.randint(0, 11)
        n = r.randint(0, 10 ** r.randint(1, 9))
        if kind == 0: return str(n)
        if kind == 1: return f"{n:,}"
        if kind == 2: return f"{n}.{r.randint(0, 999)}"
        if kind == 3: return f"${n % 5000}.{r.randint(0, 99):02d}"
        if kind == 4: return f"{r.randint(0, 25)}:{r.randint(0, 59):02d}" + r.choice(["", " AM", " pm", " p.m.", "am", ":30"])
        if kind == 5: return f"{r.randint(1, 31)}/{r.randint(1, 14)}/{r.choice([r.randint(1900, 2100), r.randint(0, 99)])}"
        if kind == 6: return f"{n % 100}{r.choice(['st', 'nd', 'rd', 'th', '%', 'kg', ' km', 'mph', 'GB', ' ft', 'x', 's', 'k', 'm'])}"
        if kind == 7: return f"{r.randint(1, 20)}/{r.randint(1, 20)}"
        if kind == 8: return f"{r.randint(1, 3000)}-{r.randint(1, 3000)}"
        if kind == 9: return f"{r.choice(['v', 'V', ''])}{r.randint(0, 20)}.{r.randint(0, 99)}.{r.randint(0, 200)}"
        if kind == 10: return f"{r.choice(['A', 'XB', 'abc', 'Q7', 'TR'])}-{n % 100000}"
        return f"{r.randint(100, 999)} {r.randint(100, 999)} {r.randint(1000, 9999)}"
    lead = ["It was", "Call", "My code is", "In", "Since", "Order", "Ticket number", "About", "The year", "He paid", "Version", "Enter",
            "At", "On", "Room", "Flight", "The RTX", "I have", "We need", "Dial", "Reference", "Due", "From", "Until", "Python", "Chapter"]
    trail = ["", " people", " dollars", " items", " today", " was fine", " or more", " and", " apples", " per cent", " to go", " times",
             " long", " ratio", " is the time", " exactly", " boxes", " miles", " in total", " = x", " percent"]
    for _ in range(420):
        text = f"{r.choice(lead)} {number()}{r.choice(trail)}{r.choice(['.', '!', '?', ',', ';', '', ' —'])}"
        if r.random() < 0.3:
            text += f" Then {number()} {r.choice(['more', 'left', 'units', 'kg', 'USD', 'pm', 'votes'])}."
        out.append(text)
    marks = list(",.?!;:\"'()—…-–") + ["...", "--", " - ", "?!", "''", "\u2019", "\u201c", "\u201d", " ", "\t", "\n", "/", "&", "@", "#", "%", "*", "_", "+", "=", "~", "<", ">", "|", "\\", "{", "}", "[", "]", "^", "$"]
    plain = "the a I you it is was of and to in that he for on with as at by an be this from or had not but what all were we when can said there use each which she do how their if will up other about out many then them these so some her would make like him into time has look two more write go see number no way could people my than first water been call who oil its now find long down day did get come made may part".split()
    for _ in range(260):
        count = r.randint(1, 9)
        text = ""
        for _ in range(count):
            w = r.choice(plain)
            if r.random() < 0.15: w = w.capitalize()
            if r.random() < 0.05: w = w.upper()
            text += w + r.choice([" "] * 6 + marks)
        out.append(text)
    return out


def fuzz(texts, count, seed):
    """Texts made of pieces of the corpus in random order: rules meet that the hand-written lines never put together."""
    r = random.Random(seed)
    pool = sorted({piece for text in texts for piece in text.split()})
    glue = [" "] * 12 + ["  ", ", ", ". ", "? ", "! ", "; ", ": ", " - ", "-", " — ", "—", "... ", " (", ") ", ' "', '" ', "'", "/", "\n", "\t", "", " & ", " @ ", " = "]
    odd = ["\u00a0", "\u2019", "\u201c", "\u2013", "\u00e9", "\u00bd", "\u00b2", "\u2026", "\u20ac", "\U0001F600", "\uff11", "\u0663", "\u00df", "\u2009", "\u00d7"]
    out = []
    for _ in range(count):
        text = ""
        for _ in range(r.randint(1, 10)):
            piece = r.choice(pool)
            roll = r.random()
            if roll < 0.04: piece = piece.upper()
            elif roll < 0.08: piece = piece.capitalize()
            elif roll < 0.10: piece = piece[:r.randint(0, len(piece))]
            elif roll < 0.12: piece = piece + r.choice(pool)
            elif roll < 0.14: piece = piece + r.choice(odd)
            elif roll < 0.16: piece = str(r.randint(0, 10 ** r.randint(1, 12)))
            text += piece + r.choice(glue)
        out.append(text)
    return out


def reference(texts):
    import dizione
    out = []
    started = time.perf_counter()
    dizione.get()
    load = time.perf_counter() - started
    started = time.perf_counter()
    for text in texts:
        try:
            result = dizione.phonemize(text)
            out.append({"ids": list(result.ids), "stress": list(result.stress)})
        except Exception as error:  # the Rust side must refuse these too
            out.append({"error": f"{type(error).__name__}: {error}"})
    first = time.perf_counter() - started
    best = first
    for _ in range(2):
        started = time.perf_counter()
        for text in texts:
            try:
                dizione.phonemize(text)
            except Exception:
                pass
        best = min(best, time.perf_counter() - started)
    return out, {"load_ms": load * 1e3, "first_pass_ms_per_line": first * 1e3 / len(texts), "warm_ms_per_line": best * 1e3 / len(texts)}


def rust_binary():
    cargo = os.environ.get("CARGO", str(Path.home() / ".cargo/bin/cargo"))
    manifest = os.environ.get("DIZIONE_MANIFEST", str(ROOT / "Cargo.toml"))
    args = [cargo, "build", "--release", "--example", "phonemize", "--manifest-path", manifest]
    if manifest == str(ROOT / "Cargo.toml"):
        args += ["-p", "dizione"]
    subprocess.run(args, check=True)
    meta = json.loads(subprocess.run([cargo, "metadata", "--no-deps", "--format-version", "1", "--manifest-path", manifest],
                                     check=True, capture_output=True, text=True).stdout)
    return Path(meta["target_directory"]) / "release/examples/phonemize"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--time", action="store_true")
    parser.add_argument("--show", type=int, default=15)
    parser.add_argument("--corpus", type=Path, help="Extra file of sentences, one per line")
    parser.add_argument("--fuzz", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    texts = corpus()
    if args.corpus:
        texts += [t for t in args.corpus.read_text(encoding="utf-8").split("\n") if t not in set(texts)]
    fixture_count = len(texts)
    if args.fuzz:
        known = set(texts)
        texts += [t for t in dict.fromkeys(fuzz(texts, args.fuzz, args.seed)) if t not in known]
    expected, python_time = reference(texts)
    binary = rust_binary()
    payload = "\n".join(json.dumps(text) for text in texts) + "\n"
    run = subprocess.run([str(binary), str(DATA), "--jsonl"], input=payload, capture_output=True, text=True)
    if run.returncode != 0:
        sys.exit(f"the Rust phonemizer failed: {run.stderr[-2000:]}")
    got = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(got) == len(texts), (len(got), len(texts))
    symbols = json.loads((ROOT / "weights/config.json").read_text(encoding="utf-8"))["vocab"]
    names = {index: symbol for symbol, index in symbols.items()}
    show = lambda r: r["error"] if "error" in r else " ".join({1: "ˈ", 2: "ˌ"}.get(s, "") + names[i] for i, s in zip(r["ids"], r["stress"]))
    bad = 0
    for text, want, have in zip(texts, expected, got):
        same = ("error" in want and "error" in have) or want == have
        if not same:
            bad += 1
            if bad <= args.show:
                print(f"MISMATCH {text!r}\n  python: {show(want)}\n  rust:   {show(have)}")
    errors = sum("error" in e for e in expected)
    print(f"{len(texts)} texts ({errors} rejected by both), {bad} mismatches")
    if args.fixture:
        path = ROOT / "crates/dizione/tests/fixture.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for text, want in list(zip(texts, expected))[:fixture_count]:
                row = {"text": text, **({"error": True} if "error" in want else want)}
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        print(f"wrote {path}")
    if args.time:
        run = subprocess.run([str(binary), str(DATA), "--time"], input=payload, capture_output=True, text=True, check=True)
        rust_time = json.loads(run.stdout)
        print("python:", json.dumps({k: round(v, 4) for k, v in python_time.items()}))
        print("rust:  ", json.dumps(rust_time))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
