import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import defaultdict

# Конфигурация
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TFIDF_DIR = os.path.join(SCRIPT_DIR, "..", "fourth")
TERMS_SUBDIR = "terms"
TOP_N_RESULTS = 10


# Загрузка данных TF-IDF из файлов
def load_tfidf_data():
    term_index = defaultdict(dict)
    doc_vectors = defaultdict(dict)

    terms_path = os.path.join(TFIDF_DIR, TERMS_SUBDIR)

    if not os.path.exists(terms_path):
        print(f"Ошибка: Папка {terms_path} не найдена!")
        print("Убедитесь, что вы выполнили Задание 4")
        exit(1)

    print(f"Загрузка данных из {terms_path}...")

    for filename in os.listdir(terms_path):
        if filename.startswith("tf_idf_terms_") and filename.endswith(".txt"):
            doc_id = filename.split("_")[-1].split(".")[0]
            with open(os.path.join(terms_path, filename), "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 3:
                        term, idf, tfidf = parts
                        term_index[term][doc_id] = float(tfidf)
                        doc_vectors[doc_id][term] = float(tfidf)

    print(f"Загружено {len(doc_vectors)} документов")
    return term_index, doc_vectors


# Обработка поискового запроса
def process_query(query, term_index):
    tokens = re.findall(r'\b[а-яё]+\b', query.lower())
    stop_words = {"и", "в", "не", "на", "с", "по", "за"}
    return [token for token in tokens if token not in stop_words and token in term_index]


# Построение вектора запроса
def build_query_vector(query_terms, term_index, all_terms):
    query_vector = np.zeros(len(all_terms))
    term_counts = defaultdict(int)

    for term in query_terms:
        term_counts[term] += 1

    query_len = max(1, len(query_terms))

    for i, term in enumerate(all_terms):
        if term in term_counts:
            # Берем IDF из первого попавшегося документа
            for doc_id in term_index[term]:
                tf = term_counts[term] / query_len
                tfidf = term_index[term][doc_id]
                query_vector[i] = tfidf / tf * tf  # TF-IDF для запроса
                break

    return query_vector


# Поиск документов
def search(query, term_index, doc_vectors, top_n=TOP_N_RESULTS):
    all_terms = sorted(term_index.keys())
    query_terms = process_query(query, term_index)

    if not query_terms:
        return []

    query_vector = build_query_vector(query_terms, term_index, all_terms)
    doc_ids = sorted(doc_vectors.keys())

    # Создаем матрицу документов
    doc_matrix = np.zeros((len(doc_ids), len(all_terms)))
    for i, doc_id in enumerate(doc_ids):
        for j, term in enumerate(all_terms):
            doc_matrix[i, j] = doc_vectors[doc_id].get(term, 0.0)

    # Вычисляем схожесть
    similarities = cosine_similarity([query_vector], doc_matrix)[0]
    results = sorted(zip(doc_ids, similarities), key=lambda x: x[1], reverse=True)

    return results[:top_n]


# Вывод результатов
def print_results(results):
    if not results:
        print("Ничего не найдено")
        return

    print("\nРезультаты поиска:")
    for i, (doc_id, score) in enumerate(results, 1):
        print(f"{i}. Документ {doc_id} (релевантность: {score:.4f})")


def main():
    print("\nВекторная поисковая система")
    print("--------------------------")

    try:
        term_index, doc_vectors = load_tfidf_data()
    except Exception as e:
        print(f"Ошибка инициализации: {str(e)}")
        exit(1)

    print("\nСистема готова к работе. Введите поисковый запрос.")
    print("Для выхода введите 'exit' или нажмите Ctrl+C\n")

    while True:
        try:
            query = input("Поиск> ").strip()
            if query.lower() in ('exit', 'quit', 'выход'):
                break

            results = search(query, term_index, doc_vectors)
            print_results(results)

        except KeyboardInterrupt:
            print("\nЗавершение работы...")
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    main()