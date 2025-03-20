import os
import requests
from bs4 import BeautifulSoup


# Функция для проверки доступности ссылки
def check_link(url):
    try:
        response = requests.get(url)
        return response.status_code == 200  # Возвращает True, если страница доступна
    except:
        return False


# Функция для сбора ссылок с lib.ru
def collect_links():
    base_url = "http://lib.ru"

    # Список авторов для поиска, изначально искала вручную
    authors = [
        "DOSTOEWSKIJ", "TOLSTOJ", "CHEHOW", "BULGAKOW", "GOGOL",
        "TURGENEW", "PUSHKIN", "LERMONTOW", "GRIBOEDOW", "GONCHAROW",
        "KUPRIN", "ANDREEW", "REMARQUE", "KAFKA", "ORWELL",
        "HEMINGWAY", "BRADBURY", "ASIMOW", "VERNE", "WELLS"
    ]

    output_file = "urls.txt"

    # Открываем файл для записи
    with open(output_file, "w", encoding="utf-8") as file:
        for author in authors:
            # Формируем URL страницы автора
            author_url = f"{base_url}/{author}/"
            print(f"Проверяем автора: {author}...")

            # Загружаем страницу автора
            response = requests.get(author_url)
            soup = BeautifulSoup(response.text, "html.parser")

            # Ищем все ссылки на тексты
            links = soup.find_all("a", href=True)
            for link in links:
                href = link["href"]
                if href.endswith(".txt"):  # Фильтруем только текстовые файлы
                    full_url = f"{base_url}/{author}/{href}"
                    if check_link(full_url):  # Проверяем доступность ссылки
                        file.write(full_url + "\n")
                        print(f"Добавлено: {full_url}")

    print(f"Ссылки сохранены в файл: {output_file}")


# Функция для скачивания страниц
def download_pages():
    # Папка для сохранения файлов
    output_folder = "downloaded_pages"
    os.makedirs(output_folder, exist_ok=True)

    with open("urls.txt", "r", encoding="utf-8") as file:
        urls = file.read().splitlines()

    index_file = open("index.txt", "w", encoding="utf-8")

    # Скачивание страниц
    for i, url in enumerate(urls, start=1):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Проверка на ошибки

            # Проверка, что страница содержит текст
            soup = BeautifulSoup(response.text, "html.parser")
            text_content = soup.get_text().strip()  # Извлечение текста

            if not text_content:
                print(f"Страница {url} не содержит текста. Пропускаем.")
                continue

            filename = os.path.join(output_folder, f"page_{i}.html")
            with open(filename, "w", encoding="utf-8") as html_file:
                html_file.write(response.text)

            index_file.write(f"{i}: {url}\n")

            print(f"Скачано: {filename}")

        except requests.exceptions.RequestException as e:
            print(f"Ошибка при скачивании {url}: {e}")

    index_file.close()
    print("Готово!")


# Основная логика
if __name__ == "__main__":
    # Сначала собираем ссылки
    collect_links()

    # Затем скачиваем страницы
    download_pages()