// Runs the deployed worker.js in Node with browser stubs: proves the import graph
// and the message protocol, including that transferring a sentence does not detach
// the buffers the whole clip is assembled from.
import { mkdirSync, cpSync, readFileSync } from "node:fs";

const sandbox = "/tmp/voca-worker-check";
mkdirSync(sandbox, { recursive: true });
cpSync("web/public/worker.js", `${sandbox}/worker.js`);
cpSync("js", `${sandbox}/js`, { recursive: true });

const messages = [];
globalThis.self = globalThis;
// A real worker detaches every buffer listed in the transfer array; mimic that.
// Lengths are recorded before the transfer, because afterwards the buffer is gone.
globalThis.postMessage = (message, transfer) => {
  const record = { ...message };
  if (message.audio) record.audioLength = message.audio.length;
  messages.push(record);
  for (const buffer of transfer || []) {
    if (buffer.byteLength === 0) throw new Error("buffer already detached");
    structuredClone(buffer, { transfer: [buffer] });
  }
};
globalThis.caches = { open: async () => null };

globalThis.fetch = async (url) => {
  if (!String(url).startsWith("https://huggingface.co")) throw new Error("unexpected fetch " + url);
  const body = readFileSync("/tmp/voca-model.gguf");
  return {
    ok: true,
    headers: new Map([["content-length", String(body.length)]]),
    async arrayBuffer() { return body.buffer.slice(body.byteOffset, body.byteOffset + body.length); },
    body: null,
  };
};

await import(`${sandbox}/worker.js`);
await globalThis.onmessage({ data: { type: "load" } });
console.log("ready:", messages.find((m) => m.type === "ready"));

messages.length = 0;
await globalThis.onmessage({ data: { type: "speak", id: 1, text: "Привет! Как дела?", voice: 0, speed: 1 } });
const done = messages.find((m) => m.type === "done");
const sentences = messages.filter((m) => m.type === "sentence");
const errors = messages.filter((m) => m.type === "error");
console.log("sentences:", sentences.length, "audio seconds:", (done.audioLength / done.sampleRate).toFixed(2));
console.log("sum of sentences:", (sentences.reduce((n, s) => n + s.audioLength, 0) / done.sampleRate).toFixed(2));
console.log("phonemes:", sentences[0].phonemes.slice(0, 40));
console.log("errors:", errors);
if (errors.length) process.exit(1);
