import re
import unittest

import _support  # noqa: F401  (adds training/ to sys.path)
from voca_ru.text.normalize import normalize


class NumbersTest(unittest.TestCase):
    def check(self, cases):
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize(source), expected)

    def test_cardinals(self):
        one = ["ноль", "один", "два", "три", "четыре", "пять", "шесть",
               "семь", "восемь", "девять", "десять", "одиннадцать",
               "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
               "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать",
               "двадцать"]
        for value, word in enumerate(one):
            with self.subTest(value=value):
                self.assertEqual(normalize(str(value)), word)
        self.check([
            ("30", "тридцать"), ("40", "сорок"), ("90", "девяносто"),
            ("100", "сто"), ("101", "сто один"), ("200", "двести"),
            ("243", "двести сорок три"), ("999", "девятьсот девяносто девять"),
            ("1000", "тысяча"), ("2000", "две тысячи"),
            ("5000", "пять тысяч"), ("21000", "двадцать одна тысяча"),
            ("1000000", "миллион"),
        ])

    def test_number_case_from_preposition(self):
        self.check([
            ("до 5", "до пяти"),
            ("от 3", "от трёх"),
            ("к 5", "к пяти"),
            ("по 2", "по двум"),
            ("из 3 книг", "из трёх книг"),
            ("без 5 минут", "без пяти минут"),
            ("в 5 часов", "в пять часов"),
            ("в 20 километрах", "в двадцати километрах"),
            ("на 5 пунктов", "на пять пунктов"),
            ("с 5 рублями", "с пятью рублями"),
        ])

    def test_ordinals(self):
        self.check([
            ("5-й", "пятый"), ("1-го", "первого"), ("20-е", "двадцатое"),
            ("3-я", "третья"), ("2026-й", "две тысячи двадцать шестой"),
            ("22-го", "двадцать второго"), ("100-й", "сотый"),
            ("40-й", "сороковой"), ("2-й", "второй"),
        ])

    def test_dates(self):
        self.check([
            ("19 ноября 2026", "девятнадцатого ноября две тысячи двадцать шестого года"),
            ("19 ноября 2026 года", "девятнадцатого ноября две тысячи двадцать шестого года"),
            ("19.11.2026", "девятнадцатого ноября две тысячи двадцать шестого года"),
            ("1 января", "первого января"),
            ("31 декабря 2025", "тридцать первого декабря две тысячи двадцать пятого года"),
        ])

    def test_years(self):
        self.check([
            ("В 2026 году", "в две тысячи двадцать шестом году"),
            ("2026 год", "две тысячи двадцать шестой год"),
            ("2026 года", "две тысячи двадцать шестого года"),
        ])

    def test_times(self):
        self.check([
            ("15:30", "пятнадцать тридцать"),
            ("9:40", "девять сорок"),
            ("14:00", "четырнадцать ноль ноль"),
            ("09:05", "девять ноль пять"),
        ])

    def test_money(self):
        self.check([
            ("1 500 ₽", "тысяча пятьсот рублей"),
            ("12 рублей 50 копеек", "двенадцать рублей пятьдесят копеек"),
            ("2 руб.", "два рубля"),
            ("5 коп.", "пять копеек"),
            ("0 копеек", "ноль копеек"),
            # a single decimal digit is tenths: 2.1$ is ten cents, not one
            ("2.1$", "два доллара десять центов"),
            ("2.10$", "два доллара десять центов"),
            ("2.01$", "два доллара один цент"),
            ("1$", "один доллар"),
            ("5,50€", "пять евро пятьдесят центов"),
            ("169.99₽", "сто шестьдесят девять рублей девяносто девять копеек"),
            ("169.9₽", "сто шестьдесят девять рублей девяносто копеек"),
        ])

    def test_units_and_percent(self):
        self.check([
            ("5 км", "пять километров"),
            ("2 км", "два километра"),
            ("1 км", "один километр"),
            ("3 кг", "три килограмма"),
            ("20%", "двадцать процентов"),
            ("1%", "один процент"),
            ("2%", "два процента"),
        ])

    def test_decimals_and_collectives(self):
        self.check([
            ("1,5 часа", "полтора часа"),
            ("1,5 минуты", "полторы минуты"),
            ("2,5", "два целых пять десятых"),
        ])

    def test_digit_by_digit(self):
        self.check([
            ("ИНН 7701234567",
             "инн семь семь ноль один два три четыре пять шесть семь"),
            ("по номеру 495 123 45 67",
             "по номеру четыре девять пять один два три четыре пять шесть семь"),
        ])


class AbbreviationsTest(unittest.TestCase):
    def test_abbreviations(self):
        cases = [
            ("т. е.", "то есть"),
            ("т.д.", "так далее"),
            ("т. к.", "так как"),
            ("см. примечание", "смотри примечание"),
            ("на с. 12", "на страница двенадцать"),
            ("ул. Пушкина", "улица пушкина"),
            ("кв. 5", "квартира пять"),
            ("5 тыс.", "пять тысяч"),
            ("2 млн", "два миллиона"),
            ("3 млрд", "три миллиарда"),
            ("№ 7", "номер семь"),
            ("др. люди", "другое люди"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize(source), expected)


class LatinAndPunctuationTest(unittest.TestCase):
    def test_latin_letter_by_letter(self):
        cases = [
            ("abc", "эй би си"),
            ("iPhone", "ай пи эйч о эн и"),
            ("ok", "о кей"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize(source), expected)

    def test_punctuation_policy(self):
        cases = [
            ("Привет...", "привет…"),
            ("«Ёлка» — да!", "«ёлка» — да!"),
            ("“Ёлка”", "«ёлка»"),
            ("цена – 5", "цена — пять"),
            ('он сказал "да"', "он сказал «да»"),
            ("1 + 2 = 3", "один два три"),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                self.assertEqual(normalize(source), expected)

    def test_yo_preserved(self):
        self.assertEqual(normalize("ЁЖ И МЁД"), "ёж и мёд")
        self.assertIn("ё", normalize("ещё"))

    def test_no_digits_or_latin(self):
        samples = [
            "19 ноября 2026 года, 15:30, 1 500 ₽, 20%, abc.",
            "ИНН 7701234567, кв. 5, 3 кг, т. е. да.",
        ]
        for text in samples:
            result = normalize(text)
            self.assertFalse(re.search(r"[0-9]", result), result)
            self.assertFalse(re.search(r"[a-z]", result), result)

    def test_idempotent(self):
        samples = [
            "Сегодня 23 января 2026 года мы встретимся в 15:30.",
            "Это стоит 1 500 ₽, скидка 20% до 30 июня.",
            "т. е. это возможно, а не обязательно; см. примечание на с. 12.",
            "Ёж съел мёд — вот это да!",
            "iPhone 15 и 2 кг яблок.",
        ]
        for text in samples:
            once = normalize(text)
            self.assertEqual(normalize(once), once)


if __name__ == "__main__":
    unittest.main()
