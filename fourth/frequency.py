import os
import math
import time
from collections import defaultdict

# Папки с данными
TOKENS_DIR = "second/tokens"
LEMMAS_DIR = "second/lemmas"
DOCS_DIR = "first/downloaded_pages"
OUTPUT_DIR = "fourth"

# Создаем папки для результатов
os.makedirs(os.path.join(OUTPUT_DIR, "terms"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "lemmas"), exist_ok=True)

print("=" * 50)
print("Начало обработки")
print(f"Входные данные: {TOKENS_DIR}, {LEMMAS_DIR}")
print(f"Выходные данные: {OUTPUT_DIR}/terms/, {OUTPUT_DIR}/lemmas/")
print("=" * 50)


# Загрузка всех токенов документов
def load_all_tokens():
    print("\n[1/6] Загрузка токенов из файлов...")
    start_time = time.time()

    doc_tokens = {}  # {doc_id: [термины]}
    all_terms = set()  # Уникальные термины
    total_docs = 0

    for token_file in os.listdir(TOKENS_DIR):
        if token_file.startswith("tokens_") and token_file.endswith(".txt"):
            doc_id = token_file.replace("tokens_", "").replace(".txt", "")
            with open(os.path.join(TOKENS_DIR, token_file), "r", encoding="utf-8") as f:
                tokens = [line.strip() for line in f if line.strip()]
                doc_tokens[doc_id] = tokens
                all_terms.update(tokens)
                total_docs += 1

    print(f"Загружено документов: {total_docs}")
    print(f"Уникальных терминов: {len(all_terms)}")
    print(f"Время загрузки: {time.time() - start_time:.2f} сек")

    return doc_tokens, all_terms, total_docs


# Загрузка всех лемм и их терминов
def load_lemmas():
    print("\n[2/6] Загрузка лемм из файлов...")
    start_time = time.time()

    lemma_to_terms = defaultdict(list)  # {лемма: [термины]}

    for lemma_file in os.listdir(LEMMAS_DIR):
        if lemma_file.startswith("lemmas_") and lemma_file.endswith(".txt"):
            with open(os.path.join(LEMMAS_DIR, lemma_file), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        lemma = parts[0]
                        terms = parts[1:]
                        lemma_to_terms[lemma].extend(terms)

    print(f"Загружено лемм: {len(lemma_to_terms)}")
    print(f"Время загрузки: {time.time() - start_time:.2f} сек")

    return lemma_to_terms


# Подсчет IDF для терминов
def compute_term_idf(doc_tokens, all_terms, total_docs):
    print("\n[3/6] Вычисление IDF для терминов...")
    start_time = time.time()

    term_idf = {}  # {термин: idf}

    for term in all_terms:
        docs_with_term = sum(1 for tokens in doc_tokens.values() if term in tokens)
        idf = math.log(total_docs / (docs_with_term + 1e-10))  # +1e-10 чтобы избежать деления на 0
        term_idf[term] = idf

    print(f"Обработано терминов: {len(term_idf)}")
    print(f"Время расчета: {time.time() - start_time:.2f} сек")

    return term_idf


# Подсчет IDF для лемм
def compute_lemma_idf(doc_tokens, lemma_to_terms, total_docs):
    print("\n[4/6] Вычисление IDF для лемм...")
    start_time = time.time()

    lemma_idf = {}  # {лемма: idf}

    for lemma, terms in lemma_to_terms.items():
        docs_with_lemma = 0
        for doc_id, tokens in doc_tokens.items():
            if any(term in tokens for term in terms):
                docs_with_lemma += 1

        idf = math.log(total_docs / (docs_with_lemma + 1e-10))
        lemma_idf[lemma] = idf

    print(f"Обработано лемм: {len(lemma_idf)}")
    print(f"Время расчета: {time.time() - start_time:.2f} сек")

    return lemma_idf


# Подсчет TF для терминов в документе
def compute_term_tf(doc_tokens, doc_id):
    term_counts = defaultdict(int)
    tokens = doc_tokens[doc_id]
    total_terms = len(tokens)

    for term in tokens:
        term_counts[term] += 1

    term_tf = {term: count / total_terms for term, count in term_counts.items()}
    return term_tf


# Подсчет TF для лемм в документе
def compute_lemma_tf(doc_tokens, doc_id, lemma_to_terms):
    lemma_counts = defaultdict(int)
    tokens = doc_tokens[doc_id]
    total_terms = len(tokens)

    for lemma, terms in lemma_to_terms.items():
        count = sum(tokens.count(term) for term in terms)
        lemma_counts[lemma] = count / total_terms

    return lemma_counts


# Сохранение TF-IDF для терминов
def save_term_tfidf(doc_id, term_tf, term_idf):
    with open(f"{OUTPUT_DIR}/terms/tf_idf_terms_{doc_id}.txt", "w", encoding="utf-8") as f:
        for term in sorted(term_tf.keys()):
            tf = term_tf[term]
            idf = term_idf[term]
            tfidf = tf * idf
            f.write(f"{term} {idf:.6f} {tfidf:.6f}\n")


# Сохранение TF-IDF для лемм
def save_lemma_tfidf(doc_id, lemma_tf, lemma_idf):
    with open(f"{OUTPUT_DIR}/lemmas/tf_idf_lemmas_{doc_id}.txt", "w", encoding="utf-8") as f:
        for lemma in sorted(lemma_tf.keys()):
            tf = lemma_tf[lemma]
            idf = lemma_idf.get(lemma, 0.0)
            tfidf = tf * idf
            f.write(f"{lemma} {idf:.6f} {tfidf:.6f}\n")


def main():
    total_start = time.time()

    # Загрузка данных
    doc_tokens, all_terms, total_docs = load_all_tokens()
    lemma_to_terms = load_lemmas()

    # Подсчет IDF
    term_idf = compute_term_idf(doc_tokens, all_terms, total_docs)
    lemma_idf = compute_lemma_idf(doc_tokens, lemma_to_terms, total_docs)

    # Обработка каждого документа
    print("\n[5/6] Расчет TF-IDF для документов...")
    doc_start = time.time()

    for i, doc_id in enumerate(doc_tokens, 1):
        # Выводим прогресс каждые 10 документов
        if i % 10 == 0 or i == len(doc_tokens):
            print(f"Обработано документов: {i}/{len(doc_tokens)}")

        # TF для терминов
        term_tf = compute_term_tf(doc_tokens, doc_id)
        save_term_tfidf(doc_id, term_tf, term_idf)

        # TF для лемм
        lemma_tf = compute_lemma_tf(doc_tokens, doc_id, lemma_to_terms)
        save_lemma_tfidf(doc_id, lemma_tf, lemma_idf)

    print(f"Время обработки документов: {time.time() - doc_start:.2f} сек")

    print("\n[6/6] Завершение...")
    print(f"Общее время выполнения: {time.time() - total_start:.2f} сек")
    print("=" * 50)
    print(f"Результаты сохранены в:\n- {OUTPUT_DIR}/terms/\n- {OUTPUT_DIR}/lemmas/")
    print("=" * 50)


if __name__ == "__main__":
    main()
