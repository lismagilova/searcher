from flask import Flask, render_template, request, abort
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from collections import defaultdict
from bs4 import BeautifulSoup
from urllib.parse import unquote
import re
from scipy.sparse import csr_matrix

app = Flask(__name__)

# Конфигурация путей
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TFIDF_DIR = os.path.join(BASE_DIR, "fourth", "terms")
DOCS_DIR = os.path.join(BASE_DIR, "first", "downloaded_pages")
TOP_N = 10


class SearchEngine:
    def __init__(self):
        self.term_index = defaultdict(dict)
        self.term_to_index = {}
        self.doc_vectors_dict = defaultdict(dict)
        self.doc_ids = []
        self.doc_titles = {}
        self.doc_contents = {}
        self.load_data()
        self.build_matrix()
        self.check_consistency()

    def load_data(self):
        print("\n" + "=" * 50)
        print("Загрузка данных поисковой системы")
        print("=" * 50)
        self.load_tfidf()
        self.load_docs_content()

    def load_tfidf(self):
        print(f"\nЗагрузка TF-IDF из {TFIDF_DIR}")
        if not os.path.exists(TFIDF_DIR):
            raise FileNotFoundError(f"Папка {TFIDF_DIR} не найдена")

        file_count = 0
        for filename in os.listdir(TFIDF_DIR):
            if filename.startswith("tf_idf_terms_") and filename.endswith(".txt"):
                file_count += 1
                doc_id = filename.split('_')[-1].split('.')[0]
                with open(os.path.join(TFIDF_DIR, filename), 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) == 3:
                            term, idf, tfidf = parts
                            self.term_index[term][doc_id] = float(tfidf)
                            self.doc_vectors_dict[doc_id][term] = float(tfidf)

        print(f"Загружено {file_count} файлов TF-IDF")

    def load_docs_content(self):
        print(f"\nЗагрузка документов из {DOCS_DIR}")
        if not os.path.exists(DOCS_DIR):
            raise FileNotFoundError(f"Папка {DOCS_DIR} не найдена")

        def clean_libru_header_footer(text):
            # Убираем типичные шапки и служебные строки с сайта Lib.ru
            patterns = [
                r'^.*?(?:Fb2\.zip|Fine HTML|Lib\.ru html).*?\n',  # служебный заголовок
                r'^.*?(?:Содержание|Printed version|КПК).*?\n',
                r'^.*?\.zip.*?\n',
            ]
            for pattern in patterns:
                text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
            return text.strip()

        file_count = 0
        for filename in os.listdir(DOCS_DIR):
            if filename.endswith(".html"):
                try:
                    doc_id = filename.split('_')[-1].split('.')[0] if filename.startswith("page_") else filename.split('.')[0]
                    with open(os.path.join(DOCS_DIR, filename), 'r', encoding='utf-8') as f:
                        content = f.read()
                        soup = BeautifulSoup(content, 'html.parser')
                        self.doc_titles[doc_id] = soup.title.string if soup.title else f"Документ {doc_id}"
                        raw_text = soup.get_text()
                        cleaned_text = clean_libru_header_footer(raw_text)
                        self.doc_contents[doc_id] = cleaned_text
                        file_count += 1
                except Exception as e:
                    print(f"Ошибка загрузки {filename}: {str(e)}")
                    continue

        print(f"Загружено {file_count} документов")

    def build_matrix(self):
        """Создание sparse-матрицы документов"""
        all_terms = sorted(self.term_index.keys())
        self.term_to_index = {term: i for i, term in enumerate(all_terms)}
        row, col, data = [], [], []

        for doc_idx, doc_id in enumerate(sorted(self.doc_vectors_dict.keys(), key=int)):
            self.doc_ids.append(doc_id)
            vector = self.doc_vectors_dict[doc_id]
            for term, tfidf in vector.items():
                row.append(doc_idx)
                col.append(self.term_to_index[term])
                data.append(tfidf)

        matrix = csr_matrix((data, (row, col)), shape=(len(self.doc_ids), len(self.term_to_index)))
        self.doc_matrix = normalize(matrix)

    def check_consistency(self):
        tfidf_docs = set(self.doc_vectors_dict.keys())
        content_docs = set(self.doc_contents.keys())

        print("\nПроверка согласованности данных:")
        print(f"Документов в TF-IDF индексе: {len(tfidf_docs)}")
        print(f"Документов с содержимым: {len(content_docs)}")

        missing_in_content = tfidf_docs - content_docs
        if missing_in_content:
            print(f"\nВНИМАНИЕ: Документы в индексе без содержимого ({len(missing_in_content)}):")
            for doc_id in sorted(missing_in_content, key=int):
                print(f" - {doc_id}")

        missing_in_index = content_docs - tfidf_docs
        if missing_in_index:
            print(f"\nВНИМАНИЕ: Документы с содержимым без индекса ({len(missing_in_index)}):")
            for doc_id in sorted(missing_in_index, key=int):
                print(f" - {doc_id}")

        is_consistent = len(missing_in_content) == 0 and len(missing_in_index) == 0
        print("\nРезультат проверки:", "OK" if is_consistent else "ЕСТЬ ПРОБЛЕМЫ")
        return is_consistent

    def search(self, query):
        query_terms = self.process_query(query)
        if not query_terms:
            return [], []

        query_vector = np.zeros(len(self.term_to_index))
        term_counts = defaultdict(int)

        for term in query_terms:
            term_counts[term] += 1

        for term in query_terms:
            if term in self.term_index:
                tf = term_counts[term] / len(query_terms)
                idfs = [self.term_index[term][doc_id] / self.doc_vectors_dict[doc_id][term]
                        for doc_id in self.term_index[term]]
                avg_idf = np.mean(idfs) if idfs else 0
                query_vector[self.term_to_index[term]] = tf * avg_idf

        query_vector = normalize(query_vector.reshape(1, -1))
        similarities = cosine_similarity(query_vector, self.doc_matrix)[0]

        results = []
        for idx in np.argsort(similarities)[::-1][:TOP_N]:
            doc_id = self.doc_ids[idx]
            results.append({
                'id': doc_id,
                'title': self.doc_titles.get(doc_id, f"Документ {doc_id}"),
                'score': round(similarities[idx], 4),
                'highlight': self.highlight_content(doc_id, query_terms)
            })

        return results, query_terms

    def highlight_content(self, doc_id, terms):
        content = self.doc_contents.get(doc_id, "")
        for term in terms:
            pattern = re.compile(re.escape(term), re.IGNORECASE)
            content = pattern.sub(f'<span class="highlight">{term}</span>', content)
        return content[:2000] + "..." if len(content) > 2000 else content

    def process_query(self, query):
        tokens = re.findall(r'\b[а-яё]+\b', query.lower())
        stop_words = {"и", "в", "не", "на", "с", "по", "за", "для", "о", "от"}
        return [t for t in tokens if t not in stop_words and t in self.term_index]


# Инициализация поисковой системы
try:
    print("\nИнициализация поисковой системы...")
    engine = SearchEngine()
    print("\nПоисковая система готова к работе!")
except Exception as e:
    print(f"\nОШИБКА ИНИЦИАЛИЗАЦИИ: {str(e)}")
    engine = None


@app.route('/', methods=['GET', 'POST'])
def index():
    if engine is None:
        return render_template('error.html',
                               message="Поисковая система не инициализирована",
                               details="Проверьте наличие данных и логи"), 500

    query = ""
    results = []
    search_terms = []

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if query:
            results, search_terms = engine.search(query)

    return render_template('search.html',
                           query=query,
                           results=results,
                           search_terms=search_terms)


@app.route('/document/<doc_id>')
def show_document(doc_id):
    if engine is None:
        abort(500, description="Поисковая система не инициализирована")

    query = unquote(request.args.get('query', ''))
    if doc_id not in engine.doc_contents:
        return render_template('not_found.html',
                               doc_id=doc_id,
                               query=query), 404

    search_terms = engine.process_query(query)
    content = engine.doc_contents[doc_id]
    title = engine.doc_titles.get(doc_id, f"Документ {doc_id}")

    for term in search_terms:
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        content = pattern.sub(f'<span class="highlight">{term}</span>', content)

    return render_template('document.html',
                           title=title,
                           content=content,
                           query=query)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
