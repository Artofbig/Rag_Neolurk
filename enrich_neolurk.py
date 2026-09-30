"""
Модуль для обогащения статей Неолурка метаданными о синонимах (редиректах).

ВАЖНО: Исходный файл базы данных (neolurk.db) НЕ МОДИФИЦИРУЕТСЯ (доступ строго в режиме Read-Only).
Все связи между редиректами и основными статьями строятся в оперативной памяти за < 1 секунды.

Возможности:
1. Анализ всех 85 000+ редиректов и разрешение цепочек перенаправлений (A -> B -> C).
2. Очистка якорей разделов (#Раздел) и нормализация заголовков.
3. Обогащение каждой статьи полем synonyms (список всех синонимов / альтернативных названий).
4. Поиск статьи как по основному названию, так и по любому из её синонимов.
5. Потоковый итератор по всем статьям для выгрузки в JSON / векторные базы без утечек памяти.
"""

import sqlite3
import time
import re
import sys
import argparse
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterator

# Импортируем очиститель викитекста из parsing.py
try:
    from parsing import clean_neolurk_wikitext
except ImportError:
    def clean_neolurk_wikitext(title: str, text: str) -> str:
        return text

# Корректное отображение русских символов в Windows
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_title(title: str) -> str:
    """Нормализует заголовок MediaWiki (заменяет _ на пробелы, удаляет невидимые символы)."""
    if not title:
        return ""
    # Удаляем невидимые символы разметки LTR/RTL (u200e и др.) и zero-width
    t = re.sub(r'[\u200e\u200f\u200b\u200c\u200d\ufeff]', '', title)
    t = t.replace("_", " ").strip()
    return t[:1].upper() + t[1:] if t else ""


def clean_redirect_target(raw_target: str) -> str:
    """Удаляет якоря секций (#...) и мусор из ссылки редиректа."""
    if not raw_target:
        return ""
    target = raw_target.split("#")[0].strip()
    return normalize_title(target)


@dataclass
class EnrichedArticle:
    id: int
    title: str
    url: str
    content: str
    synonyms: List[str]
    timestamp: str
    size: int
    
    @property
    def synonym_count(self) -> int:
        return len(self.synonyms)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "url": self.url,
            "synonyms": self.synonyms,
            "synonym_count": self.synonym_count,
            "timestamp": self.timestamp,
            "size": self.size,
            "content": self.content
        }


class NeolurkArchive:
    """
    Интерфейс для доступа к базе Неолурка в режиме Read-Only
    с автоматическим обогащением статей синонимами.
    """
    def __init__(self, db_path: str = "neolurk.db"):
        self.db_path = db_path
        # Открываем SQLite строго в режиме read-only (гарантия неизменности файла)
        self.conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.redir_pattern = re.compile(r'#(?:REDIRECT|перенаправление)\s*\[\[([^\]#|]+)', re.IGNORECASE)
        
        self.target_to_synonyms: Dict[str, List[str]] = defaultdict(list)
        self.synonym_to_target: Dict[str, str] = {}
        self._build_synonyms_index()

    def _build_synonyms_index(self):
        """Строит граф редиректов в памяти и разрешает многошаговые цепочки."""
        t0 = time.time()
        cur = self.conn.cursor()
        
        # Быстрый индексированный запрос
        cur.execute("SELECT title, redirect_target, content FROM pages WHERE is_redirect = 1")
        rows = cur.fetchall()
        
        direct_redirects = {}
        for title, target, content in rows:
            norm_title = normalize_title(title)
            final_target = None
            if target and target.strip():
                final_target = clean_redirect_target(target)
            elif content:
                m = self.redir_pattern.search(content)
                if m:
                    final_target = clean_redirect_target(m.group(1))
                    
            if norm_title and final_target and norm_title != final_target:
                direct_redirects[norm_title] = final_target

        # Разрешаем цепочки перенаправлений (если A -> B, а B -> C, то и A, и B ведут на C)
        for synonym, target in direct_redirects.items():
            curr = target
            visited = {synonym}
            for _ in range(5):  # Ограничение глубины для защиты от циклов
                if curr in direct_redirects and curr not in visited:
                    visited.add(curr)
                    curr = direct_redirects[curr]
                else:
                    break
                    
            canonical_target = curr
            self.synonym_to_target[synonym] = canonical_target
            self.target_to_synonyms[canonical_target].append(synonym)

        # Сортируем списки синонимов и удаляем дубликаты
        for k in self.target_to_synonyms:
            self.target_to_synonyms[k] = sorted(list(set(self.target_to_synonyms[k])))

        t1 = time.time()
        self.index_build_time = t1 - t0
        self.total_redirects_count = len(direct_redirects)
        self.enriched_targets_count = len(self.target_to_synonyms)

    def get_synonyms_for(self, title: str) -> List[str]:
        """Возвращает список всех синонимов для канонической статьи."""
        norm = normalize_title(title)
        canonical = self.synonym_to_target.get(norm, norm)
        return self.target_to_synonyms.get(canonical, [])

    def resolve_canonical_title(self, title_or_synonym: str) -> str:
        """По синониму возвращает настоящее каноническое название статьи."""
        norm = normalize_title(title_or_synonym)
        return self.synonym_to_target.get(norm, norm)

    def get_article(self, title_or_synonym: str, clean_markdown: bool = False) -> Optional[EnrichedArticle]:
        """
        Получает статью по её названию ИЛИ по любому из её синонимов.
        Если clean_markdown=True, поле content возвращается уже очищенным в формате Markdown.
        """
        canonical_title = self.resolve_canonical_title(title_or_synonym)
        
        cur = self.conn.cursor()
        cur.execute("""
            SELECT id, title, url, content, timestamp, size 
            FROM pages 
            WHERE is_redirect = 0 AND (title = ? OR title = ?)
            LIMIT 1
        """, (canonical_title, canonical_title.replace(" ", "_")))
        row = cur.fetchone()
        
        if not row:
            return None
            
        pid, title, url, raw_content, ts, size = row
        synonyms = self.get_synonyms_for(title)
        
        content = clean_neolurk_wikitext(title, raw_content) if clean_markdown else raw_content
        
        return EnrichedArticle(
            id=pid,
            title=title,
            url=url,
            content=content,
            synonyms=synonyms,
            timestamp=ts,
            size=size
        )

    def iter_articles(
        self,
        limit: Optional[int] = None,
        clean_markdown: bool = False,
        start_id: int = 0
    ) -> Iterator[EnrichedArticle]:
        """
        Потоковый генератор по всем полноценным статьям базы,
        обогащённым списком синонимов на лету.
        Поддерживает дозапись и продолжение с start_id.
        """
        cur = self.conn.cursor()
        query = "SELECT id, title, url, content, timestamp, size FROM pages WHERE is_redirect = 0 AND id > ? ORDER BY id ASC"
        if limit:
            query += f" LIMIT {limit}"
            
        cur.execute(query, (start_id,))
        while True:
            batch = cur.fetchmany(1000)
            if not batch:
                break
            for pid, title, url, raw_content, ts, size in batch:
                synonyms = self.get_synonyms_for(title)
                content = clean_neolurk_wikitext(title, raw_content) if clean_markdown else raw_content
                yield EnrichedArticle(
                    id=pid,
                    title=title,
                    url=url,
                    content=content,
                    synonyms=synonyms,
                    timestamp=ts,
                    size=size
                )

