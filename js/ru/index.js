// Russian text frontend: text -> ids + stress for the voca-ru GGUF.
//
//   import { readGGUF, TinyVoca } from "../tiny-voca.js";
//   import { fromGGUF } from "./index.js";
//
//   const file = readGGUF(buffer);
//   const voca = new TinyVoca(file);
//   const text = fromGGUF(file);
//   const { ids, stress } = text.phonemize("Привет, как дела?");
//   const { audio } = voca.synthesize(ids, stress, { voice: "F2" });

import { normalize } from "./normalize.js";
import { G2P, isPhone } from "./g2p.js";
import { Stressor } from "./stress.js";

const decoder = new TextDecoder();

export class RussianText {
  constructor({ vocab, stressDict, g2pLexicon, stressEndings }) {
    const symbols = Object.entries(vocab).sort((a, b) => a[1] - b[1]).map(([symbol]) => symbol);
    this.symbols = symbols;
    this.ids = new Map(symbols.map((symbol, id) => [symbol, id]));
    this.stressor = new Stressor(stressDict, stressEndings);
    this.g2p = new G2P(g2pLexicon, symbols, this.ids);
  }

  // Same fields as voca_ru.text.analyze in Python.
  analyze(text) {
    const plain = normalize(text);
    const { stressed, oov } = this.stressor.stressed(plain);
    const { symbols, ids, stress } = this.g2p.tokenize(stressed);
    return {
      plain_text: plain,
      stressed_text: stressed,
      phonemes: symbols,
      stress,
      tokens: ids,
      oov,
      phone_count: symbols.filter(isPhone).length,
    };
  }

  // What js/text.js calls: one sentence -> ids + stress (+ symbols).
  phonemize(text) {
    const { phonemes, stress, tokens } = this.analyze(text);
    return { symbols: phonemes, ids: tokens, stress };
  }
}

export function fromGGUF(file) {
  const settings = JSON.parse(decoder.decode(file.files["config.json"]));
  return new RussianText({
    vocab: settings.vocab,
    stressDict: decoder.decode(file.files["phonemizer/stress_dict.tsv"]),
    stressEndings: file.files["phonemizer/stress_endings.tsv"] === undefined
      ? ""
      : decoder.decode(file.files["phonemizer/stress_endings.tsv"]),
    g2pLexicon: decoder.decode(file.files["phonemizer/g2p_lexicon.tsv"]),
  });
}

export { normalize } from "./normalize.js";
