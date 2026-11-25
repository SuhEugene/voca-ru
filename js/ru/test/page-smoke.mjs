// Headless check of what js/index.html does: load the GGUF, split Russian text
// into sentences, phonemize and synthesize.
//
//   node js/ru/test/page-smoke.mjs [file.gguf]

import { readFileSync } from "node:fs";
import { readGGUF, TinyVoca } from "../../tiny-voca.js";
import { utterancesOf } from "../../text.js";
import { fromGGUF } from "../index.js";

const file = process.argv[2] || process.env.VOCA_GGUF ||
  "/Users/suheugene/projects/22-tiniest-voca-ru/out/voca-ru-js.gguf";
const buffer = readFileSync(file);
const voca = new TinyVoca(readGGUF(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)));
const text = fromGGUF(readGGUF(buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)));

const input = "Привет! Как дела? Сегодня хорошая погода, и я очень рад. Позвони мне в 15:30.";
const utterances = utterancesOf(input, text, voca.config.max_tokens - 12);
console.assert(utterances.length === 4, `expected 4 sentences, got ${utterances.length}`);

let seconds = 0;
for (const [seed, { ids, stress, phonemes }] of utterances.entries()) {
  console.assert(ids.length === stress.length, "ids and stress lengths differ");
  console.assert(ids.length > 2 && ids.length <= voca.config.max_tokens, `unexpected ids length ${ids.length}`);
  console.assert(stress.some((level) => level === 1), `no stressed vowel in ${phonemes}`);
  const { audio, durations } = voca.synthesize(ids, stress, { voice: "F2", seed });
  console.assert(audio.length > 0, "empty audio");
  console.assert(audio.every(Number.isFinite), "audio has non-finite samples");
  console.assert(durations.length === ids.length, "one duration per id");
  const clip = audio.length / voca.config.sample_rate;
  console.assert(clip > 0.3 && clip < 30, `implausible clip length ${clip}s`);
  seconds += clip;
  console.log(`sentence ${seed + 1}: ${ids.length} ids, ${phonemes}`);
}

console.log(`ok: ${utterances.length} sentences, ${seconds.toFixed(2)} s of audio at ${voca.config.sample_rate} Hz, voice F2`);
