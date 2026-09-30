"""
Улучшенный модуль для очистки и конвертации викитекста Неолурка (MediaWiki) в чистый Markdown.

Исправлены ключевые проблемы прототипа:
1. Вложенные скобки в изображениях: [[Файл:...[[ссылка]]...]] больше не оставляет мусорные "]]" в тексте.
2. Вложенные шаблоны: цитаты {{Q|...{{nobr|...}}...|Автор}} больше не обрезаются и корректно форматируют автора.
3. Таблицы: синтаксис MediaWiki-таблиц ({| ... |}) корректно очищается от CSS-стилей, а полезный текст сохраняется.
4. Списки: нумерованные списки MediaWiki (# пункт) больше не ломают разметку заголовков Markdown (# Заголовок 1).
5. Заголовки: устойчивость к пробелам и комментариям на концах строк (== Заголовок == ).
6. Ссылки: корректная обработка внешних ссылок [http://site.com Описание] -> [Описание](http://site.com).
7. HTML: декодирование сущностей (&quot;, &nbsp; и др.), удаление комментариев <!-- ... -->, тегов <gallery> и очистка разметки.
"""

import sqlite3
import re
import html
import sys

# Обеспечиваем корректный вывод UTF-8 в консоли Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def remove_wiki_images(text: str) -> str:
    """
    Удаляет теги файлов и картинок MediaWiki ([[Файл:...]], [[Изображение:...]])
    с корректным учётом любой глубины вложенных ссылок в подписях.
    """
    prefixes = ("[[файл:", "[[изображение:", "[[file:", "[[image:")
    result = []
    i = 0
    n = len(text)
    
    while i < n:
        lower_slice = text[i:i + 15].lower()
        if any(lower_slice.startswith(p) for p in prefixes):
            depth = 0
            j = i
            while j < n:
                if text[j:j + 2] == "[[":
                    depth += 1
                    j += 2
                elif text[j:j + 2] == "]]":
                    depth -= 1
                    j += 2
                    if depth <= 0:
                        break
                else:
                    j += 1
            if j < n and text[j] == "\n":
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
            
    return "".join(result)


def process_single_template(template_body: str) -> str:
    """Обрабатывает один плоский (самый внутренний) шаблон MediaWiki."""
    parts = [p.strip() for p in template_body.split('|')]
    if not parts:
        return ""
    
    name = parts[0].lower()
    args = parts[1:]
    
    # Зачёркнутый текст: {{s|текст}} -> ~~текст~~
    if name in ('s', 'strike', 'del', 'зачёркнутый', 'зачеркнуто'):
        for arg in args:
            if '=' not in arg:
                return f"~~{arg}~~"
        return ""
        
    # Неразрывный текст: {{nobr|текст}} -> текст
    if name in ('nobr', 'nowrap'):
        for arg in args:
            if '=' not in arg:
                return arg
        return ""
        
    # Ссылки на Википедию / классический Лурк: {{w|Статья|Текст}} -> Текст
    if name in ('w', 'wikipedia', 'википедия', 'lurk', 'lm', 'неолурк', 'neolurk'):
        pos_args = [a for a in args if '=' not in a]
        if len(pos_args) >= 2:
            return pos_args[1]
        elif len(pos_args) == 1:
            return pos_args[0]
        return ""

    # Спойлеры: {{spoiler|текст}} или {{spoiler|Заголовок|текст}}
    if name in ('spoiler', 'спойлер'):
        pos_args = [a for a in args if '=' not in a]
        if pos_args:
            return pos_args[-1]
        return ""

    # Цитаты: {{q|текст|автор}} или {{цитата|текст|автор=...}}
    if name in ('q', 'quote', 'цитата', 'высказывание', 'citat'):
        quote_text = ""
        author = ""
        pos_args = []
        for a in args:
            if a.lower().startswith('автор='):
                author = a.split('=', 1)[1].strip()
            elif a.lower().startswith('источник='):
                pass
            elif a.lower().startswith('pre='):
                continue
            elif '=' in a:
                continue
            else:
                pos_args.append(a)
                
        if pos_args:
            quote_text = pos_args[0]
            if len(pos_args) > 1 and not author:
                author = pos_args[1]
                
        if not quote_text:
            return ""
            
        lines = [line.strip() for line in quote_text.splitlines() if line.strip()]
        md_quote = '\n' + '\n'.join(f'> {line}' for line in lines)
        if author:
            md_quote += f'\n> — *{author}*'
        return md_quote + '\n'

    # Языковые шаблоны: {{lang|en|word}}
    if name == 'lang' and len(args) >= 2:
        return args[1]
    if name.startswith('lang-') and args:
        return args[0]
        
    # Комментарии / всплывающие подсказки: {{comment|текст|подсказка}}
    if name in ('comment', 'подсказка') and args:
        return args[0]

    # Служебные плашки (стабы, навигационные списки, плашки плагиата) удаляем
    return ""


def unwind_templates(text: str, max_depth: int = 25) -> str:
    """Раскрывает шаблоны от самых внутренних к внешним во избежание обрезки."""
    pattern = re.compile(r'\{\{([^{}]+)\}\}')
    for _ in range(max_depth):
        if not pattern.search(text):
            break
        text = pattern.sub(lambda m: process_single_template(m.group(1)), text)
        
    # Удаляем остаточные незакрытые шаблоны
    text = re.sub(r'\{\{[^{}]*$', '', text)
    text = re.sub(r'^\s*\}\}', '', text, flags=re.MULTILINE)
    return text


def clean_tables(text: str) -> str:
    """
    Очищает таблицы MediaWiki ({| ... |}), извлекая полезное содержимое
    ячеек и удаляя служебные CSS/HTML атрибуты оформления.
    """
    def table_replacer(match):
        table_raw = match.group(0)
        lines = table_raw.splitlines()
        body_lines = []
        
        for line in lines[1:-1]:
            line = line.strip()
            if not line:
                continue
            if line.startswith('|-'):
                body_lines.append('')
                continue
            if line.startswith(('|', '!')):
                content = line[1:].strip()
                # Удаляем атрибуты вида style="...", width=50% | Текст ячейки
                if '|' in content and not content.startswith('{'):
                    parts = content.split('|')
                    if any(attr in parts[0].lower() for attr in ('class=', 'style=', 'bgcolor=', 'colspan=', 'rowspan=', 'align=', 'width=')):
                        content = '|'.join(parts[1:]).strip()
                if content:
                    body_lines.append(content)
            else:
                body_lines.append(line)
                
        return '\n' + '\n'.join(body_lines) + '\n'

    return re.sub(r'\{\|.*?\n\|\}', table_replacer, text, flags=re.DOTALL)


def convert_headers(text: str) -> str:
    """Преобразует заголовки == Header == в Markdown ## Header устойчиво к концевым пробелам."""
    def header_replacer(match):
        level = len(match.group(1))
        title = match.group(2).strip()
        md_level = min(level, 6)
        return f"{'#' * md_level} {title}"
        
    pattern = re.compile(r'^[ \t]*(={2,6})[ \t]*(.*?)[ \t]*\1[ \t]*(?:<!--.*?-->)?[ \t]*$', re.MULTILINE)
    return pattern.sub(header_replacer, text)


def convert_lists(text: str) -> str:
    """Конвертирует списки MediaWiki в корректный Markdown."""
    lines = text.splitlines()
    res = []
    
    for line in lines:
        stripped = line.strip()
        # Вложенный нумерованный список
        if stripped.startswith('##'):
            content = stripped[2:].strip()
            res.append(f"    1. {content}")
        # Нумерованный список MediaWiki (# пункт или #пункт)
        elif stripped.startswith('#'):
            content = stripped[1:].strip()
            res.append(f"1. {content}")
        # Вложенный маркированный список
        elif stripped.startswith('**'):
            content = stripped[2:].strip()
            res.append(f"  * {content}")
        # Определение ; Термин : Описание
        elif stripped.startswith(';') and ':' in stripped:
            term, desc = stripped[1:].split(':', 1)
            res.append(f"**{term.strip()}**: {desc.strip()}")
        # Отступ через двоеточие (ответ/цитата)
        elif stripped.startswith(':'):
            content = stripped[1:].strip()
            res.append(f"> {content}")
        else:
            res.append(line)
            
    return '\n'.join(res)


def clean_neolurk_wikitext(title: str, text: str) -> str:
    """
    Полная очистка викитекста статьи Неолурка и конвертация в чистый Markdown.
    """
    if not text:
        return f"# {title}\n"
        
    # 0. Нормализация переводов строк и пробелов
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.replace('\xa0', ' ')
    
    # 1. Отрезаем служебные разделы в конце (Примечания, Ссылки, См. также и т.д.)
    text = re.split(
        r'\n==+\s*(?:Примечания|Ссылки|См\.\s*также|Источники|Литература|Галерея)\s*==+.*',
        text,
        flags=re.IGNORECASE | re.DOTALL
    )[0]
    
    # 2. Удаляем комментарии <!-- ... -->
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 3. Удаляем сноски <ref>...</ref> и одиночные <ref ... />
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<ref[^>]*/>', '', text, flags=re.IGNORECASE)
    
    # 4. Удаляем блоки галерей <gallery>...</gallery>
    text = re.sub(r'<gallery[^>]*>.*?</gallery>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # 5. Удаляем категории [[Категория:...]] и [[Category:...]]
    text = re.sub(r'\[\[(?:Категория|Category):[^\]]+\]\]\n?', '', text, flags=re.IGNORECASE)
    
    # 6. Удаляем файлы и изображения с учетом вложенных скобок
    text = remove_wiki_images(text)
    
    # 7. Очистка таблиц {| ... |} с извлечением цитат и текста
    text = clean_tables(text)
    
    # 8. Раскрытие шаблонов от внутренних к внешним
    text = unwind_templates(text)
    
    # 9. Внешние ссылки [http://site.com Описание] -> [Описание](http://site.com)
    text = re.sub(r'\[(https?://[^\s\]]+)\s+([^\]]+)\]', r'[\2](\1)', text)
    text = re.sub(r'\[(https?://[^\s\]]+)\]', r'\1', text)
    
    # 10. Внутренние вики-ссылки [[Статья|Текст]] -> Текст
    def link_replacer(match):
        link_content = match.group(1).strip()
        if '|' in link_content:
            return link_content.split('|')[-1].strip()
        if link_content.startswith('#'):
            return link_content[1:].strip()
        return link_content

    text = re.sub(r'\[\[([^\]]+)\]\]', link_replacer, text)
    
    # 11. Базовые HTML-теги форматирования в Markdown
    text = re.sub(r'<(?:s|strike|del)>(.*?)</(?:s|strike|del)>', r'~~\1~~', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<(?:b|strong)>(.*?)</(?:b|strong)>', r'**\1**', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<(?:i|em)>(.*?)</(?:i|em)>', r'*\1*', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<code>(.*?)</code>', r'`\1`', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<hr\s*/?>', '\n---\n', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    
    # Декодирование HTML-сущностей (&quot;, &laquo;, &nbsp; и т.д.)
    text = html.unescape(text)
    
    # 12. Конвертируем списки MediaWiki (# -> 1., ** -> *) до заголовков!
    text = convert_lists(text)
    
    # 13. Конвертируем заголовки в Markdown (== -> ##)
    text = convert_headers(text)
    
    # 14. Конвертируем жирный шрифт и курсив MediaWiki
    text = re.sub(r"'''''(.*?)'''''", r"***\1***", text)
    text = re.sub(r"'''(.*?)'''", r"**\1**", text)
    text = re.sub(r"''(.*?)''", r"*\1*", text)
    
    # 15. Чистка оставшихся мусорных скобок и лишних пустых строк
    text = re.sub(r'\]\]+', '', text)
    text = re.sub(r'\}\}+', '', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    
    return f"# {title}\n\n{text}"


if __name__ == "__main__":
    # Пример использования и проверка на статье "Владимир Медейко"
    conn = sqlite3.connect("neolurk.db")
    cursor = conn.cursor()

    article_title = "Владимир Медейко"
    cursor.execute("SELECT id, title, content FROM pages WHERE title = ?", (article_title,))
    row = cursor.fetchone()

    if row:
        page_id, title, raw_content = row
        print(f"[*] Статья найдена: ID {page_id}, '{title}'")
        print(f"[*] Исходная длина: {len(raw_content)} символов")
        
        cleaned_markdown = clean_neolurk_wikitext(title, raw_content)
        print(f"[*] Длина после очистки: {len(cleaned_markdown)} символов")
        print("\n" + "=" * 60)
        print("РЕЗУЛЬТАТ ОЧИСТКИ (первые 1500 символов):")
        print("=" * 60)
        print(cleaned_markdown[:1500])
        print("...\n" + "=" * 60)
    else:
        print(f"[!] Статья '{article_title}' не найдена в neolurk.db.")
        
    conn.close()
