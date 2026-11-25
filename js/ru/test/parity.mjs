// Compares the JS Russian frontend with a dump of Python voca_ru.text.analyze.
//
//   node js/ru/test/parity.mjs [file.gguf] [fixtures/parity.json]
//
// The fixture was written outside the worktree with the repo venv (see
// js/README.md). Reports exact-token parity overall, on sentences whose every
// word is in both shipped lexicons, and on sentences with OOV words.

import { readFileSync } from "node:fs";
import { readGGUF } from "../../tiny-voca.js";
import { fromGGUF } from "../index.js";
import { normalize } from "../normalize.js";

const file = process.argv[2] || process.env.VOCA_GGUF ||
  "/Users/suheugene/projects/22-tiniest-voca-ru/out/voca-ru-js.gguf";
const fixture = process.argv[3] || new URL("fixtures/parity.json", import.meta.url).pathname;

const buffer = readFileSync(file);
const text = fromGGUF(readGGUF(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)));
const rows = JSON.parse(readFileSync(fixture, "utf8"));

const WORDS = /[а-яё]+(?:-[а-яё]+)*/g;
function covered(sentence) {
  for (const [word] of normalize(sentence).matchAll(WORDS)) {
    for (const part of word.split("-")) {
      if (!text.stressor.dictionary.has(part) || !text.g2p.lexicon.has(part)) return false;
    }
  }
  return true;
}

const same = (a, b) => a.length === b.length && a.every((value, i) => value === b[i]);
const stats = { all: [0, 0, 0], both: [0, 0, 0], oov: [0, 0, 0] };
const failures = [];
for (const [number, row] of rows.entries()) {
  const got = text.analyze(row.text);
  const ids = same(got.tokens, row.tokens), stress = same(got.stress, row.stress);
  const bucket = covered(row.text) ? "both" : "oov";
  for (const key of ["all", bucket]) {
    stats[key][0]++;
    if (ids) stats[key][1]++;
    if (ids && stress) stats[key][2]++;
  }
  if (!ids || !stress) {
    let at = 0;
    while (at < Math.max(got.tokens.length, row.tokens.length) &&
           got.tokens[at] === row.tokens[at] && got.stress[at] === row.stress[at]) at++;
    failures.push({ number, bucket, row, got, at, ids, stress });
  }
}

const percent = (part, whole) => (whole ? (100 * part / whole).toFixed(2) : "0.00");
for (const [label, [total, ids, both]] of Object.entries(stats)) {
  console.log(`${label}: ${total} sentences, ids equal ${ids} (${percent(ids, total)}%), ids and stress equal ${both} (${percent(both, total)}%)`);
}

for (const { number, bucket, row, got, at, ids } of failures.slice(0, 10)) {
  console.log(`\n#${number} (${bucket}, ${row.set}) ${JSON.stringify(row.text)}`);
  if (!ids) {
    console.log(`  token ${at}: python ${row.tokens[at]} (${row.phonemes?.[at] ?? "?"}) js ${got.tokens[at]} (${got.phonemes[at]})`);
    console.log(`  plain python ${JSON.stringify(row.plain)}`);
    console.log(`  plain js     ${JSON.stringify(got.plain_text)}`);
  } else {
    console.log(`  stress differs at ${at}`);
    console.log(`  python ${JSON.stringify(row.stressed)}\n  js     ${JSON.stringify(got.stressed_text)}`);
  }
}
if (failures.length > 10) console.log(`\n… ${failures.length - 10} more failures`);

const [total, ids, both] = stats.both;
if (ids !== total) process.exitCode = 1;
console.log(`\nboth lexicons: ${ids}/${total} ids, ${both}/${total} ids+stress; failures: ${failures.length}/${rows.length}`);
