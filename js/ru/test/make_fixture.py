"""Regenerate js/ru/test/fixtures/parity.json with the Python frontend.

Run from the main repo, with the shared venv (the JS worktree has no data/):

    . scripts/cache-env.sh
    VOCA_CACHE=$PWD/.cache .cache/venvs/r4/bin/python \
        worktrees/original/js/ru/test/make_fixture.py [repo-root]

It reads data/scripts-ru/batch{1,2}.jsonl and training/voca_ru, and writes the
fixture next to this file. Nothing is downloaded.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else
                       os.environ.get("VOCA_RU_ROOT", "/Users/suheugene/projects/22-tiniest-voca-ru"))
sys.path.insert(0, os.path.join(ROOT, "training"))

from voca_ru.text import analyze  # noqa: E402
from voca_ru.text.stress import Stressor  # noqa: E402

SYNTHETIC = [
    "Привет! Как дела?",
    "Сегодня 19 ноября 2026 года мы встретимся в 15:30.",
    "Это стоит 1 500 ₽, скидка 20% до 30 июня.",
    "Купи 2 кг яблок и 1,5 литра молока.",
    "В 2026 году исполнится 100 лет.",
    "Цена — 99,90 руб., доставка 300 рублей.",
    "Позвони по номеру 495 123 45 67.",
    "Он родился 1 января 2000 года.",
    "С 5 мая по 10 июня идут дожди.",
    "До двух тысяч человек пришли на митинг.",
    "У неё ёлка, ёж и мёд.",
    "iPhone 15 стоит 1000 долларов.",
    "Т. е. это так, и т. д.",
    "5-й вагон, 2-я платформа.",
    "Скидка 15 % на всё.",
    "Около 3 км пути.",
    "Мы жили в 5 км от города.",
    "На 20 страницах — 100 ошибок.",
    "Он сказал: «Да, конечно».",
    "Цена выросла на 12,5 процента.",
    "Заплатил 250 копеек.",
    "Встреча в 09:05 утра.",
    "Это было 12.04.1961.",
    "Пробежал 42,2 км.",
    "Мне 30 лет, а брату 21 год.",
    "2800 человек, 5000 рублей.",
    "Смотри стр. 5 и рис. 3.",
    "Живу на ул. Пушкина, д. 7, кв. 12.",
    "Вес — 1 000 000 тонн.",
    "Температура −5 градусов.",
]


def main():
    rows = []
    for name in ("batch1", "batch2"):
        path = os.path.join(ROOT, "data", "scripts-ru", f"{name}.jsonl")
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                rows.append(json.loads(line))
    first, second = rows[:4500], rows[4500:]
    picked = [first[i] for i in range(0, len(first), max(1, len(first) // 190))][:190]
    picked += [second[i] for i in range(0, len(second), max(1, len(second) // 160))][:160]

    stressor = Stressor().load()
    fixture = []
    for item in picked:
        fixture.append({"set": "batch", "text": item["text"]})
    for sentence in SYNTHETIC:
        fixture.append({"set": "synthetic", "text": sentence})

    for item in fixture:
        result = analyze(item["text"], stressor=stressor)
        item.update({
            "plain": result["plain_text"],
            "stressed": result["stressed_text"],
            "tokens": result["tokens"],
            "stress": result["stress"],
            "oov": result["oov"],
        })

    out = os.path.join(HERE, "fixtures", "parity.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(fixture, handle, ensure_ascii=False, separators=(",", ":"))
    print(f"{out}: {len(fixture)} sentences, {sum(f['oov'] for f in fixture)} with OOV words")


if __name__ == "__main__":
    main()
