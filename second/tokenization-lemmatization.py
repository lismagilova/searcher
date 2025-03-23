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
                  "как", "что", "так", "же", "же"}  # Убираем предлоги и союзы
    return set(word for word in words if word not in stop_words)


# Функция для лемматизации и группировки по леммам
def lemmatize(tokens):
    morph = pymorphy2.MorphAnalyzer()
    lemma_dict = {}

    for token in tokens:
        lemma = morph.parse(token)[0].normal_form  # Получаем лемму
        if lemma not in lemma_dict:
            lemma_dict[lemma] = []
        lemma_dict[lemma].append(token)

    return lemma_dict


# Папка с HTML-файлами
input_folder = "first/downloaded_pages"
all_tokens = set()

# Обрабатываем все файлы в папке
for filename in os.listdir(input_folder):
    if filename.endswith(".html"):
        with open(os.path.join(input_folder, filename), "r", encoding="utf-8") as file:
            raw_text = file.read()
            clean_text = clean_html(raw_text)
            tokens = tokenize(clean_text)
            all_tokens.update(tokens)

# Лемматизируем все токены
lemma_dict = lemmatize(all_tokens)

# Сохраняем список токенов
with open("second/tokens.txt", "w", encoding="utf-8") as file:
    for token in sorted(all_tokens):
        file.write(token + "\n")

# Сохраняем список лемм с токенами
with open("second/lemmas.txt", "w", encoding="utf-8") as file:
    for lemma, words in lemma_dict.items():
        file.write(lemma + " " + " ".join(sorted(set(words))) + "\n")

print("Файлы tokens.txt и lemmas.txt успешно созданы.")
