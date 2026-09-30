"""
Скрипт для парсинга всех страниц с сайта Неолурк (neolurk.org) в базу данных SQLite.

Использует официальный MediaWiki API (/w/api.php) с пакетной загрузкой (по 50 статей за запрос),
что в сотни раз быстрее и надёжнее обычного парсинга HTML через BeautifulSoup.

Особенности:
- Не требует сторонних библиотек (работает на стандартной библиотеке Python 3: urllib, sqlite3, json).
- Поддерживает возобновление (докачку): если прервать (Ctrl+C) или оборвётся интернет, 
  скрипт продолжит с того же места.
- Автоматические повторные попытки при сбоях сети (exponential backoff).
- Сохраняет: id, название (title), url, исходный текст (wikitext), признак редиректа, цель редиректа, дату правки и размер.
- Настройка задержки (delay), лимита страниц, пропуска редиректов.
"""

import sys
import time
import json
import sqlite3
import argparse
import urllib.request
import urllib.error
import urllib.parse
import re
import signal

# Корректное отображение русских символов в консоли Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

API_ENDPOINT = "https://neolurk.org/w/api.php"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"


def init_db(db_path: str) -> sqlite3.Connection:
    """Инициализация базы данных и создание таблиц и индексов."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    
    # Таблица для статей
    cur.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            id INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            namespace INTEGER NOT NULL,
            url TEXT NOT NULL,
            content TEXT,
            is_redirect INTEGER DEFAULT 0,
            redirect_target TEXT,
            timestamp TEXT,
            size INTEGER,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Индексы для быстрого поиска
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_title ON pages(title)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_is_redirect ON pages(is_redirect)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pages_namespace ON pages(namespace)")
    
    # Таблица состояния для сохранения токена продолжения (continue token)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS crawler_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    
    conn.commit()
    return conn


def save_state(conn: sqlite3.Connection, continue_params: dict):
    """Сохранение состояния пагинации для докачки."""
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO crawler_state (key, value) VALUES (?, ?)",
        ("continue_params", json.dumps(continue_params))
    )
    conn.commit()


def load_state(conn: sqlite3.Connection) -> dict:
    """Загрузка сохранённого токена пагинации."""
    cur = conn.cursor()
    cur.execute("SELECT value FROM crawler_state WHERE key = ?", ("continue_params",))
    row = cur.fetchone()
    if row and row[0]:
        try:
            return json.loads(row[0])
        except Exception:
            return {}
    return {}


def clear_state(conn: sqlite3.Connection):
    """Сброс состояния докачки (для запуска с нуля)."""
    cur = conn.cursor()
    cur.execute("DELETE FROM crawler_state WHERE key = ?", ("continue_params",))
    conn.commit()


def api_request(params: dict, max_retries: int = 5, retry_delay: float = 3.0) -> dict:
    """Выполнение HTTP-запроса к MediaWiki API с обработкой ошибок и повторами."""
    params["format"] = "json"
    query_string = urllib.parse.urlencode(params)
    url = f"{API_ENDPOINT}?{query_string}"
    
    req = urllib.request.Request(
        url,
        headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
        "Referer": "https://neolurk.org/",
        "Origin": "https://neolurk.org",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-origin"
            }
    )
    
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.info().get("Content-Encoding") == "gzip":
                    import gzip
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ConnectionResetError) as e:
            if attempt == max_retries:
                raise
            print(f"\n[!] Ошибка сети ({e}). Повтор {attempt}/{max_retries} через {retry_delay:.1f}с...")
            time.sleep(retry_delay)
            retry_delay *= 1.5


def get_site_statistics() -> dict:
    """Получение статистики с сайта (общее количество статей, правок и т.д.)."""
    try:
        data = api_request({"action": "query", "meta": "siteinfo", "siprop": "statistics"})
        return data.get("query", {}).get("statistics", {})
    except Exception:
        return {}


def parse_redirect_target(content: str) -> str:
    """Извлечение названия целевой статьи для редиректа."""
    if not content:
        return None
    m = re.search(r"#(?:REDIRECT|перенаправление)\s*\[\[([^\]]+)\]\]", content, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def run_scraper(
    db_path: str = "neolurk.db",
    namespace: int = 0,
    batch_size: int = 50,
    delay: float = 0.5,
    limit: int = None,
    skip_redirects: bool = False,
    reset: bool = False
):
    """Основной процесс парсинга и сохранения."""
    conn = init_db(db_path)
    
    if reset:
        print("[*] Сброс состояния: начинаем парсинг с первой страницы.")
        clear_state(conn)
        continue_params = {}
    else:
        continue_params = load_state(conn)
        if continue_params:
            print(f"[*] Возобновление с предыдущей позиции: {continue_params.get('gapcontinue', continue_params)}")
    
    # Запрос актуальной статистики сайта
    stats = get_site_statistics()
    if stats:
        print(f"[*] Статистика Неолурка:")
        print(f"    - Энциклопедических статей: {stats.get('articles', 'н/д')}")
        print(f"    - Всего страниц в базе:      {stats.get('pages', 'н/д')}")
    
    print(f"[*] База данных SQLite: {db_path}")
    print(f"[*] Пространство имён:   {namespace} ({'основные статьи' if namespace == 0 else f'ID {namespace}'})")
    print(f"[*] Размер пакета:       {batch_size} стр./запрос")
    print(f"[*] Пауза между запросами: {delay} сек.")
    if limit:
        print(f"[*] Лимит сохранения:   {limit} стр.")
    print("=" * 65)
    
    total_saved = 0
    total_skipped = 0
    start_time = time.time()
    stop_requested = False
    
    def signal_handler(sig, frame):
        nonlocal stop_requested
        print("\n\n[!] Остановка по Ctrl+C. Завершаем запись текущей пачки...")
        stop_requested = True
        
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not stop_requested:
            params = {
                "action": "query",
                "generator": "allpages",
                "gapnamespace": namespace,
                "gaplimit": batch_size,
                "prop": "revisions|info",
                "rvprop": "content|timestamp",
                "inprop": "url"
            }
            
            if continue_params:
                params.update(continue_params)
            
            res = api_request(params)
            query = res.get("query", {})
            pages = query.get("pages", {})
            
            if not pages:
                print("\n[*] Страницы не найдены либо достигнут конец базы.")
                clear_state(conn)
                break
            
            records_to_insert = []
            last_title = ""
            
            for pid_str, page in pages.items():
                pid = int(pid_str)
                title = page.get("title", "")
                last_title = title
                ns = page.get("ns", namespace)
                url = page.get("fullurl", f"https://neolurk.org/wiki/{urllib.parse.quote(title)}")
                is_redir = 1 if "redirect" in page else 0
                
                revs = page.get("revisions", [])
                content = revs[0].get("*", "") if revs else ""
                timestamp = revs[0].get("timestamp", "") if revs else ""
                size = page.get("length", len(content.encode("utf-8")))
                
                redir_target = parse_redirect_target(content) if is_redir else None
                
                if skip_redirects and is_redir:
                    total_skipped += 1
                    continue
                
                records_to_insert.append((
                    pid,
                    title,
                    ns,
                    url,
                    content,
                    is_redir,
                    redir_target,
                    timestamp,
                    size
                ))
            
            # Пакетная вставка в SQLite
            cur = conn.cursor()
            cur.executemany("""
                INSERT OR REPLACE INTO pages (
                    id, title, namespace, url, content, is_redirect, redirect_target, timestamp, size
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, records_to_insert)
            
            total_saved += len(records_to_insert)
            
            # Сохранение токена пагинации
            if "continue" in res:
                continue_params = res["continue"]
                save_state(conn, continue_params)
            else:
                conn.commit()
                print("\n\n[+] Парсинг успешно завершён! Все страницы выгружены.")
                clear_state(conn)
                break
            
            conn.commit()
            
            # Прогресс
            elapsed = time.time() - start_time
            rate = total_saved / elapsed if elapsed > 0 else 0
            short_title = (last_title[:25] + "..") if len(last_title) > 25 else last_title
            sys.stdout.write(
                f"\r[+] Сохранено: {total_saved} | Пропущено редиректов: {total_skipped} | "
                f"Скорость: {rate:.1f} стр/сек | Текущая: '{short_title}'"
            )
            sys.stdout.flush()
            
            if limit and total_saved >= limit:
                print(f"\n[*] Достигнут лимит в {limit} страниц.")
                break
                
            time.sleep(delay)
            
    except Exception as e:
        print(f"\n[!] Ошибка в процессе парсинга: {e}")
        conn.commit()
        raise
    finally:
        conn.close()
        elapsed = time.time() - start_time
        print(f"\n{'=' * 65}")
        print(f"[*] Результат:")
        print(f"    - Сохранено страниц в БД: {total_saved}")
        print(f"    - Пропущено редиректов:   {total_skipped}")
        print(f"    - Время работы:           {elapsed:.1f} сек.")
        print(f"    - Файл базы данных:       {db_path}")
        print(f"{'=' * 65}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Парсер всех статей сайта Неолурк (neolurk.org) в SQLite"
    )
    parser.add_argument(
        "--db",
        type=str,
        default="neolurk.db",
        help="Путь к файлу базы данных SQLite (по умолчанию: neolurk.db)"
    )
    parser.add_argument(
        "--namespace",
        type=int,
        default=0,
        help="Пространство имён MediaWiki (0 = основные энциклопедические статьи)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Количество страниц за один сетевой запрос (рекомендуется 50)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Задержка между запросами в секундах (по умолчанию: 0.5)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Ограничить количество страниц (например, для проверки: --limit 100)"
    )
    parser.add_argument(
        "--skip-redirects",
        action="store_true",
        help="Не сохранять страницы-перенаправления (редиректы)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Сбросить прогресс и начать сбор сначала"
    )
    
    args = parser.parse_args()
    run_scraper(
        db_path=args.db,
        namespace=args.namespace,
        batch_size=args.batch_size,
        delay=args.delay,
        limit=args.limit,
        skip_redirects=args.skip_redirects,
        reset=args.reset
    )
