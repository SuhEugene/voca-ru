// Runs the model (GGUF in pure JS) and the Russian frontend off the main thread.
// Messages in: {type: "load"} and {type: "speak", id, text, voice, speed}.
// Messages out: status, ready, sentence (with its audio, as soon as it is made),
// done (with the whole clip), error.
import { readGGUF, TinyVoca } from "./js/tiny-voca.js";
import { fromGGUF } from "./js/ru/index.js";
import { utterancesOf, limited } from "./js/text.js";

const MODEL = "https://huggingface.co/SuhEugene/voca-ru/resolve/main/gguf/voca-ru-js.gguf";

let voca = null, phonemizer = null, starting = null;
const status = (text, progress) => postMessage({ type: "status", text, progress });

// The model is kept in the browser's cache: a second visit does not touch the network.
async function download(url, label) {
  const store = await caches.open("voca-ru-v5").catch(() => null);
  if (store) {
    const hit = await store.match(url);
    if (hit) { status(label, 1); return new Uint8Array(await hit.arrayBuffer()); }
  }
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return new Uint8Array(await response.arrayBuffer());
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    status(label, received / total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  if (store) store.put(url, new Response(bytes)).catch(() => {});
  return bytes;
}

async function start() {
  status("Скачиваю модель", 0);
  const bytes = await download(MODEL, "Скачиваю модель");
  status("Читаю файл");
  const file = readGGUF(bytes.buffer);
  voca = new TinyVoca(file);
  phonemizer = fromGGUF(file);
  postMessage({ type: "ready", sampleRate: voca.config.sample_rate, voices: voca.voices });
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === "load") {
      starting ||= start();
      await starting;
      return;
    }
    if (data.type !== "speak") return;
    starting ||= start();
    await starting;

    const { id, text, voice, speed } = data;
    const began = performance.now();
    const parts = utterancesOf(text, phonemizer, voca.config.max_tokens - 12);
    const phonemeTime = performance.now() - began;
    if (!parts.length) throw new Error("в тексте нечего произносить");

    const rate = voca.config.sample_rate;
    const pieces = [];
    let total = 0;
    parts.forEach(({ ids, stress, phonemes }, index) => {
      const audio = limited(voca.synthesize(ids, stress, { voice, speed, seed: index }).audio);
      pieces.push(audio);
      total += audio.length;
      // transfer a copy: transferring `audio` itself detaches the buffer we still
      // need to assemble the whole clip below
      const copy = audio.slice();
      postMessage({ type: "sentence", id, index, total: parts.length, phonemes, audio: copy, sampleRate: rate },
                  [copy.buffer]);
    });

    const whole = new Float32Array(total);
    pieces.reduce((at, piece) => (whole.set(piece, at), at + piece.length), 0);
    postMessage({ type: "done", id, audio: whole, sampleRate: rate,
                  totalTime: performance.now() - began, phonemeTime }, [whole.buffer]);
  } catch (error) {
    postMessage({ type: "error", message: error.message || String(error) });
  }
};
