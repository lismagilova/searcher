import os
import re
import json
from collections import defaultdict


# Загружаем леммы из файлов в папке
def load_lemmas(lemmas_dir):
    lemma_to_terms = defaultdict(set)

    for lemma_file in os.listdir(lemmas_dir):
        if lemma_file.startswith('lemmas_') and lemma_file.endswith('.txt'):
            with open(os.path.join(lemmas_dir, lemma_file), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) > 1:
                        lemma = parts[0]
                        terms = parts[1:]
                        lemma_to_terms[lemma].update(terms)

    return lemma_to_terms


# Строим инвертированный индекс по файлам токенов
def build_inverted_index(tokens_dir):
    inverted_index = defaultdict(set)
    doc_ids = []

    for token_file in os.listdir(tokens_dir):
        if token_file.startswith('tokens_') and token_file.endswith('.txt'):
            doc_id = len(doc_ids)
            doc_ids.append(token_file)

            with open(os.path.join(tokens_dir, token_file), 'r', encoding='utf-8') as f:
                for line in f:
                    term = line.strip()
                    if term:
                        inverted_index[term].add(doc_id)

    return inverted_index, doc_ids


# Сохраняем инвертированный индекс в файл
def save_inverted_index(inverted_index, doc_ids, filename='third/inverted_index.json'):
    # Создаем папку third, если она не существует
    os.makedirs('third', exist_ok=True)

    # Преобразуем множества в списки для сериализации
    serializable_index = {
        term: sorted(doc_ids_set)  # Сортируем для воспроизводимости
        for term, doc_ids_set in inverted_index.items()
    }

    data = {
        'inverted_index': serializable_index,
        'doc_ids': doc_ids
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# Загружаем инвертированный индекс из файла
def load_inverted_index(filename='third/inverted_index.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Проверяем структуру файла
        if not isinstance(data, dict) or 'inverted_index' not in data or 'doc_ids' not in data:
            print(f"Файл {filename} имеет неверный формат")
            return None, None

        # Преобразуем списки обратно в множества
        inverted_index = defaultdict(set)
        for term, doc_ids_list in data['inverted_index'].items():
            inverted_index[term].update(doc_ids_list)

        return inverted_index, data['doc_ids']
    except FileNotFoundError:
        print(f"Файл {filename} не найден")
        return None, None
    except json.JSONDecodeError:
        print(f"Файл {filename} поврежден или имеет неверный формат")
        return None, None
    except Exception as e:
        print(f"Ошибка при загрузке индекса: {e}")
        return None, None


# Разбираем запрос, заменяя леммы на соответствующие термины
def parse_query(query, lemma_to_terms):
    # Обрабатываем операторы и скобки
    tokens = re.findall(r'\(|\)|AND|OR|NOT|\b\w+\b', query.upper())

    processed_tokens = []
    for token in tokens:
        if token in ('AND', 'OR', 'NOT', '(', ')'):
            processed_tokens.append(token)
        else:
            # Ищем термины по лемме
            lemma = token.lower()
            if lemma in lemma_to_terms:
                terms = list(lemma_to_terms[lemma])
                if len(terms) > 1:
                    processed_tokens.append('(' + ' OR '.join(terms) + ')')
                else:
                    processed_tokens.append(terms[0])
            else:
                processed_tokens.append(lemma)

    return ' '.join(processed_tokens)


# Выполняем булев поиск по запросу
def evaluate_query(query, inverted_index, doc_ids):

    # Преобразуем запрос в обратную польскую нотацию
    def shunting_yard(tokens):
        precedence = {'NOT': 3, 'AND': 2, 'OR': 1}
        output = []
        operators = []

        for token in tokens:
            if token == '(':
                operators.append(token)
            elif token == ')':
                while operators[-1] != '(':
                    output.append(operators.pop())
                operators.pop()
            elif token in precedence:
                while (operators and operators[-1] != '(' and
                       precedence[operators[-1]] >= precedence[token]):
                    output.append(operators.pop())
                operators.append(token)
            else:
                output.append(token)

        while operators:
            output.append(operators.pop())

        return output

    # Вычисляем результат
    def evaluate_postfix(postfix):
        stack = []

        for token in postfix:
            if token == 'AND':
                right = stack.pop()
                left = stack.pop()
                stack.append(left & right)
            elif token == 'OR':
                right = stack.pop()
                left = stack.pop()
                stack.append(left | right)
            elif token == 'NOT':
                operand = stack.pop()
                all_docs = set(range(len(doc_ids)))
                stack.append(all_docs - operand)
            else:
                stack.append(inverted_index.get(token.lower(), set()))

        return stack.pop() if stack else set()

    tokens = re.findall(r'\(|\)|AND|OR|NOT|\b\w+\b', query.upper())
    postfix = shunting_yard(tokens)
    result_docs = evaluate_postfix(postfix)

    return [doc_ids[doc_id] for doc_id in sorted(result_docs)]


def main():
    # Загрузка данных
    lemmas_dir = 'second/lemmas'  # Папка с леммами
    tokens_dir = 'second/tokens'  # Папка с токенами

    # Пытаемся загрузить сохраненный индекс
    inverted_index, doc_ids = load_inverted_index()

    if inverted_index is None:
        print("Строим инвертированный индекс...")
        lemma_to_terms = load_lemmas(lemmas_dir)
        inverted_index, doc_ids = build_inverted_index(tokens_dir)
        save_inverted_index(inverted_index, doc_ids)
        print("Инвертированный индекс сохранен в third/inverted_index.json")
    else:
        print("Загружен сохраненный инвертированный индекс")
        lemma_to_terms = load_lemmas(lemmas_dir)

    print(f"Загружено {len(doc_ids)} документов")
    print(f"Индекс содержит {len(inverted_index)} уникальных терминов")
    print(f"Загружено {len(lemma_to_terms)} лемм")

    # Интерактивный поиск
    while True:
        query = input("\nВведите поисковый запрос (или 'exit' для выхода):\n> ")
        if query.lower() == 'exit':
            break

        try:
            # Преобразуем леммы в запросе в термины
            expanded_query = parse_query(query, lemma_to_terms)
            print(f"Расширенный запрос: {expanded_query}")

            # Выполняем поиск
            results = evaluate_query(expanded_query, inverted_index, doc_ids)

            # Выводим результаты
            print(f"\nНайдено документов: {len(results)}")
            for doc in results:
                print(f"- {doc}")

        except Exception as e:
            print(f"Ошибка обработки запроса: {e}")


if __name__ == "__main__":
    main()