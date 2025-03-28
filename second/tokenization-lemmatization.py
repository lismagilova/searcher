import re
import html
import os
from bs4 import BeautifulSoup
import pymorphy2


# Функция для очистки текста от HTML-разметки
def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text()
    return html.unescape(text)


# Функция токенизации текста
def tokenize(text):
    words = re.findall(r'\b[а-яА-ЯёЁ]+\b', text.lower())  # Извлекаем только слова
    stop_words = {"и", "в", "во", "не", "на", "с", "по", "за", "из", "от", "до", "под", "у", "о", "об", "а", "но",
                  "как", "что", "так", "же"}  # Убираем предлоги и союзы
    return set(word for word in words if word not in stop_words)


# Функция для лемматизации и группировки по леммам
def lemmatize(tokens):
    morph = pymorphy2.MorphAnalyzer()
    lemma_dict = {}

    for token in tokens:
        lemma = morph.parse(token)[0].normal_form  # Получаем лемму
        if lemma not in lemma_dict:
            lemma_dict[lemma] = []
        if token not in lemma_dict[lemma]:  # Убираем дубликаты токенов внутри леммы
            lemma_dict[lemma].append(token)

    return lemma_dict


# Папка с HTML-файлами
input_folder = "first/downloaded_pages"
output_folder = "second"

# Создаем папки для токенов и лемм
os.makedirs(os.path.join(output_folder, "tokens"), exist_ok=True)
os.makedirs(os.path.join(output_folder, "lemmas"), exist_ok=True)

# Обрабатываем каждый файл отдельно
for idx, filename in enumerate(os.listdir(input_folder), start=1):
    if filename.endswith(".html"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as file:
            raw_text = file.read()
            clean_text = clean_html(raw_text)
            tokens = tokenize(clean_text)

        # Лемматизируем токены
        lemma_dict = lemmatize(tokens)

        # Сохраняем токены в файл (один токен на строку)
        tokens_file = os.path.join(output_folder, "tokens", f"tokens_{idx}.txt")
        with open(tokens_file, "w", encoding="utf-8") as file:
            for token in sorted(tokens):
                file.write(token + "\n")

        # Сохраняем леммы и токены в файл (одна лемма + токены на строку)
        lemmas_file = os.path.join(output_folder, "lemmas", f"lemmas_{idx}.txt")
        with open(lemmas_file, "w", encoding="utf-8") as file:
            for lemma, words in lemma_dict.items():
                file.write(lemma + " " + " ".join(sorted(words)) + "\n")

print("Обработка завершена. Файлы сохранены.")