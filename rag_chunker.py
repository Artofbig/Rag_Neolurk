"""
Модуль для RAG-чанкинга статей Неолурка на базе LangChain.

Архитектура: Section-Aware Parent-Child (Иерархический чанкинг по разделам).

Почему Parent-Child — лучший выбор для энциклопедических данных:
1. Решает проблему размытия эмбеддингов: поиск идёт по компактным Child-чанкам (300-500 символов),
   что даёт высокую точность совпадения цитат, сленга и фактов.
2. Решает проблему нехватки контекста у LLM: найденный Child-чанк ссылается на Parent-документ
   (весь логический раздел целиком, 1500-2500 символов), который и передаётся в языковую модель.
3. Учитывает структуру Вики: чанки привязаны к заголовкам Markdown (## Раздел), предотвращая
   разрыв предложений и склейку несвязанных тем.
4. Контекстный префикс (Contextual Retrieval): каждый дочерний чанк обогащается строкой
   с каноническим названием статьи, списком синонимов и текущим разделом.
"""

import sys
import json
import argparse
from typing import List, Tuple, Dict, Any, Optional, Iterator

# LangChain компоненты
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)

# Модуль доступа к базе с синонимами
from enrich_neolurk import NeolurkArchive, EnrichedArticle

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class NeolurkHierarchicalChunker:
    """
    Иерархический чанкер для RAG.
    
    Делит каждую статью на:
    - Parent Documents (для генерации LLM): логические разделы Markdown.
    - Child Chunks (для векторного поиска): небольшие смысловые фрагменты с метаданными.
    """
    def __init__(
        self,
        parent_chunk_size: int = 1800,
        parent_chunk_overlap: int = 150,
        child_chunk_size: int = 400,
        child_chunk_overlap: int = 60,
        add_context_prefix_to_child: bool = True
    ):
        self.parent_chunk_size = parent_chunk_size
        self.parent_chunk_overlap = parent_chunk_overlap
        self.child_chunk_size = child_chunk_size
        self.child_chunk_overlap = child_chunk_overlap
        self.add_context_prefix_to_child = add_context_prefix_to_child

        # Сплиттер заголовков Markdown
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=False
        )

        # Сплиттер для разбивки слишком длинных разделов на родительские блоки
        self.parent_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Сплиттер для дочерних поисковых чанков
        self.child_text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""]
        )

    def chunk_article(self, article: EnrichedArticle) -> Tuple[List[Document], List[Document]]:
        """
        Разбивает одну статью на списки (parents, children).
        """
        parent_docs: List[Document] = []
        child_docs: List[Document] = []

        synonyms_str = ", ".join(article.synonyms) if article.synonyms else ""

        # Базовые метаданные статьи (для фильтрации, цитирования и поиска)
        base_meta = {
            "doc_id": article.id,
            "title": article.title,
            "url": article.url,
            "synonyms": article.synonyms,
            "synonyms_str": synonyms_str,
            "synonym_count": article.synonym_count,
            "timestamp": article.timestamp,
        }

        # 1. Деление по логическим секциям Markdown (## Раздел)
        raw_sections = self.markdown_splitter.split_text(article.content)
        if not raw_sections:
            raw_sections = [Document(page_content=article.content, metadata={})]

        parent_idx = 0

        for sec in raw_sections:
            # Формируем путь раздела: Статья > Раздел > Подраздел
            h_parts = [sec.metadata.get(f"h{lvl}") for lvl in (1, 2, 3, 4) if sec.metadata.get(f"h{lvl}")]
            section_breadcrumb = " > ".join(h_parts) if h_parts else article.title

            sec_content = sec.page_content.strip()
            if not sec_content:
                continue

            # 2. Формирование Parent документов
            if len(sec_content) <= self.parent_chunk_size:
                parent_texts = [sec_content]
            else:
                parent_texts = self.parent_text_splitter.split_text(sec_content)

            for p_text in parent_texts:
                parent_idx += 1
                parent_id = f"art_{article.id}_p{parent_idx}"

                parent_metadata = {
                    **base_meta,
                    "parent_id": parent_id,
                    "section": section_breadcrumb,
                    "chunk_type": "parent",
                    "char_count": len(p_text),
                }

                parent_doc = Document(
                    page_content=p_text,
                    metadata=parent_metadata
                )
                parent_docs.append(parent_doc)

                # 3. Формирование Child документов из текущего Parent
                child_texts = self.child_text_splitter.split_text(p_text)
                for child_idx, c_text in enumerate(child_texts, 1):
                    child_id = f"{parent_id}_c{child_idx}"

                    # Contextual Retrieval prefix:
                    # Добавляем контекст статьи, чтобы изолированный абзац легко находился по эмбеддингам
                    if self.add_context_prefix_to_child:
                        context_header = f"[Статья: {article.title}"
                        if synonyms_str:
                            context_header += f" | Синонимы: {synonyms_str}"
                        context_header += f" | Раздел: {section_breadcrumb}]\n"
                        child_page_content = context_header + c_text
                    else:
                        child_page_content = c_text

                    child_metadata = {
                        **base_meta,
                        "chunk_id": child_id,
                        "parent_id": parent_id,
                        "section": section_breadcrumb,
                        "chunk_type": "child",
                        "child_index": child_idx,
                        "char_count": len(c_text),
                    }

                    child_doc = Document(
                        page_content=child_page_content,
                        metadata=child_metadata
                    )
                    child_docs.append(child_doc)

        return parent_docs, child_docs

    def chunk_articles_stream(
        self,
        articles: Iterator[EnrichedArticle]
    ) -> Iterator[Tuple[List[Document], List[Document]]]:
        """Потоковая обработка статей."""
        for art in articles:
            yield self.chunk_article(art)
if __name__ == "__main__":
    a = NeolurkArchive()
    g = a.get_article("МКАД", clean_markdown=True)
    b = NeolurkHierarchicalChunker()
    print(b.chunk_article(g))