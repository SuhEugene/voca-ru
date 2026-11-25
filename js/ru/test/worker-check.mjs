// Runs the deployed worker.js in Node with browser stubs: proves the import graph
// and the message protocol, not the browser APIs themselves.
import { mkdirSync, cpSync, copyFileSync, readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";

const sandbox = "/tmp/voca-worker-check";
mkdirSync(sandbox, { recursive: true });
cpSync("web/public/worker.js", `${sandbox}/worker.js`);
cpSync("js", `${sandbox}/js`, { recursive: true });

const messages = [];
globalThis.self = globalThis;
globalThis.postMessage = (message, transfer) => messages.push(message);
globalThis.caches = { open: async () => null };

const realFetch = globalThis.fetch;
globalThis.fetch = async (url) => {
  if (String(url).startsWith("https://huggingface.co")) {
    const body = readFileSync("/tmp/voca-model.gguf");
    return {
      ok: true,
      headers: new Map([["content-length", String(body.length)]]),
      async arrayBuffer() { return body.buffer.slice(body.byteOffset, body.byteOffset + body.length); },
      body: null,
    };
  }
  return realFetch(url);
};

await import(`${sandbox}/worker.js`);
await globalThis.onmessage({ data: { type: "load" } });
console.log("ready:", messages.find((m) => m.type === "ready"));

messages.length = 0;
await globalThis.onmessage({ data: { type: "speak", id: 1, text: "Привет! Как дела?", voice: 0, speed: 1 } });
const done = messages.find((m) => m.type === "done");
const sentences = messages.filter((m) => m.type === "sentence");
console.log("sentences:", sentences.length, "audio seconds:", (done.audio.length / done.sampleRate).toFixed(2));
console.log("phonemes:", sentences[0].phonemes.slice(0, 60));
console.log("error messages:", messages.filter((m) => m.type === "error"));
