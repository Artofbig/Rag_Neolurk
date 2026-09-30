"""
Единый модуль RAG для Неолурка:
1. Hybrid Retrieval: Dense (FAISS) + Lexical (BM25) со слиянием через Reciprocal Rank Fusion (RRF).
2. Reranker: Cross-Encoder (BAAI/bge-reranker-base) с порогом отсечения релевантности (score_threshold).
   Если порог не набран — система честно сообщает, что не знает ответа.
3. Parent-Child: поиск по гранулярным Child-чанкам, передача в LLM цельного Parent-раздела.
4. Генерация: локальная LLM через Ollama (по умолчанию Qwen 2.5: qwen2.5:7b).
5. Модели эмбеддингов с поддержкой русского языка:
   - 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2' (по умолчанию: лёгкая, 384d, ~470 МБ)
   - 'intfloat/multilingual-e5-base' (768d)
   - 'deepvk/USER-bge-m3' (1024d)
   - 'cointegrated/rubert-tiny2' (312d, сверхлёгкая)
"""

import os
import sys
import re
import gc
import json
import time
import math
import pickle
import argparse
from typing import List, Dict, Any, Optional, Tuple

import faiss
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore

try:
    import bm25s
except ImportError:
    bm25s = None

try:
    from langchain_ollama import ChatOllama
except ImportError:
    ChatOllama = None

from enrich_neolurk import NeolurkArchive, EnrichedArticle
from rag_chunker import NeolurkHierarchicalChunker

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Модели
RUSSIAN_EMBEDDING_MODELS = {
    "minilm": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "e5": "intfloat/multilingual-e5-base",
    "bge-m3": "deepvk/USER-bge-m3",
    "tiny2": "cointegrated/rubert-tiny2",
    "giga": "ai-sage/Giga-Embeddings-instruct-480M-0826",
}

RUSSIAN_RERANKER_MODELS = {
    "dity": "DiTy/cross-encoder-russian-msmarco",   # Сверхлёгкий, быстрый, специально для русского языка (~117 МБ)
    "bge": "BAAI/bge-reranker-base",                 # Мультиязычный бейзлайн (~1.1 ГБ)
}

DEFAULT_EMBED_MODEL = RUSSIAN_EMBEDDING_MODELS["minilm"]
DEFAULT_RERANKER_MODEL = RUSSIAN_RERANKER_MODELS["dity"]

RUSSIAN_LLM_MODELS = {
    "qwen-3b": "Qwen/Qwen2.5-3B-Instruct",      # Быстрый, отлично знает русский, ~6 ГБ VRAM в FP16
    "qwen-1.5b": "Qwen/Qwen2.5-1.5B-Instruct",  # Сверхлёгкий, ~3 ГБ VRAM
    "qwen-7b": "Qwen/Qwen2.5-7B-Instruct",      # Максимальное качество, ~14 ГБ VRAM
}
DEFAULT_LLM_MODEL = RUSSIAN_LLM_MODELS["qwen-3b"]
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b"
DEFAULT_INDEX_DIR = "faiss_index"



def normalize_score(s: float) -> float:
    """
    Нормализация скора реранкера в диапазон [0.0, 1.0].
    Если модель уже возвращает вероятности (например, DiTy cross-encoder с sigmoid),
    значение сохраняется как есть. Если возвращаются сырые логиты (например, BGE),
    применяется сигмоида.
    """
    if s < 0.0 or s > 1.0:
        try:
            return 1.0 / (1.0 + math.exp(-s))
        except OverflowError:
            return 0.0 if s < 0 else 1.0
    return float(s)


def tokenize_ru(text: str) -> List[str]:
    """Лёгкий токенизатор для русского и английского текста (для BM25)."""
    return re.findall(r"[a-zA-Zа-яА-Я0-9ёЁ]+", text.lower())


def get_russian_stemmer():
    """Стеммер для русского языка для точного лексического поиска по словоформам."""
    try:
        from nltk.stem.snowball import SnowballStemmer
        stem = SnowballStemmer("russian").stem
        return lambda words: [stem(w) for w in words]
    except Exception:
        def _light_stem(w):
            return re.sub(
                r"(ами|ями|ов|ев|ей|ам|ям|ом|ем|ой|ей|ия|ии|ие|ий|ью|ья|ье|ый|ой|ий|ая|ое|ые|ых|их|ую|юю|а|е|и|о|у|ы|ь|я|ю)$",
                "",
                w.lower()
            )
        return lambda words: [_light_stem(w) for w in words]


class BM25SIndex:
    """Высокопроизводительный лексический ретривер на базе C/NumPy (bm25s)."""
    def __init__(self, retriever: Any, corpus: List[Dict[str, Any]]):
        self.retriever = retriever
        self.corpus = corpus
        self.stemmer = get_russian_stemmer()

    def search(self, query: str, k: int = 10) -> List[Document]:
        """Поиск по запросу с русской токенизацией и стеммингом (время отклика ~2-5 мс)."""
        if bm25s is None:
            return []
        q_tokens = bm25s.tokenize([query], stopwords="ru", stemmer=self.stemmer, show_progress=False)
        results, scores = self.retriever.retrieve(
            q_tokens,
            corpus=self.corpus,
            k=min(k, len(self.corpus)),
            show_progress=False
        )
        docs = []
        for doc_item, score in zip(results[0], scores[0]):
            if score <= 0:
                continue
            meta = dict(doc_item.get("metadata", {}))
            meta["bm25_score"] = float(score)
            docs.append(Document(
                page_content=doc_item.get("page_content", ""),
                metadata=meta
            ))
        return docs

    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)
        self.retriever.save(folder_path, corpus=self.corpus)

    @classmethod
    def load(cls, folder_path: str) -> "BM25SIndex":
        if bm25s is None:
            raise ImportError("Пакет 'bm25s' не установлен. Установите: pip install bm25s")
        retriever = bm25s.BM25.load(folder_path, load_corpus=True)
        return cls(retriever=retriever, corpus=retriever.corpus)


def get_embeddings(
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 512
) -> HuggingFaceEmbeddings:
    """Инициализация модели эмбеддингов с автоматическим выбором GPU и FP16."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except (ImportError, Exception):
        device = "cpu"

    model_kwargs = {"device": device, "trust_remote_code": True}
    if device == "cuda":
        model_kwargs["model_kwargs"] = {"torch_dtype": torch.float16}

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"normalize_embeddings": True, "batch_size": batch_size}
    )


class Reranker:
    """Кросс-энкодер реранкер для точной оценки релевантности пар (запрос, документ)."""
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        from sentence_transformers import CrossEncoder
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except (ImportError, Exception):
            device = "cpu"
        print(f"[*] Загрузка модели реранкера: {model_name} (device: {device})...")
        self.model = CrossEncoder(model_name, device=device)

    def compute_scores(self, query: str, texts: List[str]) -> List[float]:
        """Возвращает нормализованные скоры (0.0 - 1.0) для списка текстов."""
        if not texts:
            return []
        pairs = [[query, text] for text in texts]
        raw_scores = self.model.predict(pairs)
        return [normalize_score(float(s)) for s in raw_scores]


def load_docstore(docstore_path: str) -> Dict[str, Any]:
    """
    Безопасная загрузка docstore.json с автоматическим исправлением обрыва
    в случае внезапной остановки процесса или сбоя ядра.
    """
    if not os.path.exists(docstore_path):
        raise FileNotFoundError(f"Файл {docstore_path} не найден.")
    try:
        with open(docstore_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[!] Предупреждение: {docstore_path} был прерван при записи ({e}). Выполняем авто-восстановление...")
        with open(docstore_path, "rb") as f:
            raw_bytes = f.read()
        raw_text = raw_bytes.decode("utf-8", errors="ignore").rstrip()
        last_brace = raw_text.rfind("}}")
        if last_brace != -1:
            raw_text = raw_text[:last_brace + 2] + "\n}"
        else:
            last_single = raw_text.rfind("}")
            raw_text = raw_text[:last_single + 1] + "\n}"
        docstore = json.loads(raw_text)
        print(f"[✔] docstore.json успешно восстановлен! Загружено разделов: {len(docstore):,}")
        tmp_fixed = docstore_path + ".tmp"
        with open(tmp_fixed, "w", encoding="utf-8") as f:
            json.dump(docstore, f, ensure_ascii=False)
        os.replace(tmp_fixed, docstore_path)
        return docstore


class HybridNeolurkRetriever:
    """
    Гибридный ретривер (Dense FAISS + Lexical BM25 + RRF)
    с опциональным Cross-Encoder реранкингом и порогом релевантности.
    Работает как в гибридном режиме (FAISS + BM25), так и в чисто плотном (только FAISS).
    """
    def __init__(
        self,
        vectorstore: FAISS,
        bm25: Optional[BM25Retriever],
        docstore: Dict[str, Dict[str, Any]],
        model_name: str = DEFAULT_EMBED_MODEL
    ):
        self.vectorstore = vectorstore
        self.bm25 = bm25
        self.docstore = docstore
        self.model_name = model_name
        self._reranker: Optional[Reranker] = None

    def get_reranker(self, model_name: str = DEFAULT_RERANKER_MODEL) -> Reranker:
        """Ленивая загрузка реранкера (только когда он требуется)."""
        if self._reranker is None:
            self._reranker = Reranker(model_name)
        return self._reranker

    def save(self, folder_path: str = DEFAULT_INDEX_DIR):
        os.makedirs(folder_path, exist_ok=True)
        self.vectorstore.save_local(folder_path)
        if self.bm25 is not None:
            if hasattr(self.bm25, "save"):
                self.bm25.save(os.path.join(folder_path, "bm25s"))
            else:
                try:
                    with open(os.path.join(folder_path, "bm25.pkl"), "wb") as f:
                        pickle.dump(self.bm25, f)
                except Exception as e:
                    print(f"[!] Не удалось сериализовать legacy BM25 ({e})")
        tmp_docstore = os.path.join(folder_path, "docstore.json.tmp")
        docstore_file = os.path.join(folder_path, "docstore.json")
        with open(tmp_docstore, "w", encoding="utf-8") as f:
            json.dump(self.docstore, f, ensure_ascii=False)
        os.replace(tmp_docstore, docstore_file)
        print(f"[✔] Индекс успешно сохранён в '{folder_path}'")


    @classmethod
    def load(
        cls,
        folder_path: str = DEFAULT_INDEX_DIR,
        embeddings: Optional[HuggingFaceEmbeddings] = None
    ) -> "HybridNeolurkRetriever":
        chk_path = os.path.join(folder_path, "checkpoint.json")
        model_name = DEFAULT_EMBED_MODEL
        if os.path.exists(chk_path):
            try:
                with open(chk_path, "r", encoding="utf-8") as f:
                    chk = json.load(f)
                model_name = chk.get("model_name", DEFAULT_EMBED_MODEL)
                print(f"[*] Загрузка индекса из '{folder_path}':")
                print(f"    - Модель эмбеддингов:  {model_name}")
                print(f"    - Статей в индексе:    {chk.get('articles_processed', 'n/a')} (последний ID: {chk.get('last_page_id', 'n/a')})")
                print(f"    - Векторов в FAISS:    {chk.get('faiss_total', 'n/a')}")
                print(f"    - Статус индексации:   {chk.get('status', 'in_progress')}")
            except Exception:
                pass

        if embeddings is None:
            embeddings = get_embeddings(model_name)

        vectorstore = FAISS.load_local(
            folder_path=folder_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )

        bm25 = None
        bm25s_dir = os.path.join(folder_path, "bm25s")
        bm25_file = os.path.join(folder_path, "bm25.pkl")
        if bm25s is not None and os.path.exists(bm25s_dir):
            try:
                bm25 = BM25SIndex.load(bm25s_dir)
                print(f"    - Лексический поиск:   BM25S (C/NumPy, быстрый)")
            except Exception as e:
                print(f"[!] Ошибка загрузки BM25S: {e}")
        elif os.path.exists(bm25_file):
            try:
                with open(bm25_file, "rb") as f:
                    bm25 = pickle.load(f)
                print(f"    - Лексический поиск:   rank_bm25 (legacy pkl)")
            except Exception as e:
                print(f"[!] Ошибка загрузки BM25: {e}")

        docstore = load_docstore(os.path.join(folder_path, "docstore.json"))

        return cls(vectorstore=vectorstore, bm25=bm25, docstore=docstore, model_name=model_name)


    def search_candidates(
        self,
        query: str,
        k_dense: int = 10,
        k_bm25: int = 10,
        rrf_k: int = 60
    ) -> List[Document]:
        """Первичный отбор кандидатов (Dense FAISS + Lexical BM25 со слиянием через RRF)."""
        dense_query = query
        if "giga" in self.model_name.lower():
            dense_query = f"Instruct: Дан вопрос, найди подходящий фрагмент статьи из энциклопедии Неолурк.\nQuery: {query}"
        elif "e5" in self.model_name.lower():
            dense_query = f"query: {query}"

        dense_results = self.vectorstore.similarity_search(dense_query, k=k_dense)

        # Если BM25 ещё не построен, возвращаем результаты dense поиска
        if self.bm25 is None:
            return dense_results

        if hasattr(self.bm25, "search"):
            bm25_results = self.bm25.search(query, k=k_bm25)
        elif hasattr(self.bm25, "invoke"):
            self.bm25.k = k_bm25
            bm25_results = self.bm25.invoke(query)
        else:
            bm25_results = []

        rrf_scores: Dict[str, float] = {}
        candidate_map: Dict[str, Document] = {}

        for rank, doc in enumerate(dense_results, start=1):
            pid = doc.metadata.get("parent_id", doc.metadata.get("chunk_id", str(id(doc))))
            if pid not in candidate_map:
                candidate_map[pid] = doc
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (0.5 / (rrf_k + rank))

        for rank, doc in enumerate(bm25_results, start=1):
            pid = doc.metadata.get("parent_id", doc.metadata.get("chunk_id", str(id(doc))))
            if pid not in candidate_map:
                candidate_map[pid] = doc
            rrf_scores[pid] = rrf_scores.get(pid, 0.0) + (0.5 / (rrf_k + rank))

        sorted_pids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)
        return [candidate_map[pid] for pid in sorted_pids]

    def search_and_rerank(
        self,
        query: str,
        top_parents: int = 3,
        k_candidates: int = 15,
        score_threshold: float = 0.35,
        rerank: bool = True,
        reranker_model: str = DEFAULT_RERANKER_MODEL
    ) -> Tuple[List[Document], bool]:
        """
        Полный цикл поиска:
        1. Гибридный отбор топ кандидатов (FAISS + BM25).
        2. Реранкинг через Cross-Encoder.
        3. Проверка score_threshold: если скор ниже порога, возвращает ([], False).
        4. Разрешение Child -> Parent (цельный раздел для LLM без дубликатов).
        
        Возвращает: (список_parent_документов, пройден_ли_порог)
        """
        candidates = self.search_candidates(query, k_dense=k_candidates, k_bm25=k_candidates)
        if not candidates:
            return [], False

        # Если реранкинг включен
        if rerank:
            reranker = self.get_reranker(reranker_model)
            texts = [c.page_content for c in candidates]
            scores = reranker.compute_scores(query, texts)

            # Привязываем скор реранкера к чанкам
            scored_candidates = list(zip(candidates, scores))
            # Сортируем по убыванию скора реранкера
            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            # Проверяем лучший скор на соответствие порогу
            best_score = scored_candidates[0][1] if scored_candidates else 0.0
            if best_score < score_threshold:
                # Порог не пройден — релевантной информации нет
                return [], False

            # Оставляем только те, что преодолели порог
            filtered_candidates = [
                (doc, score) for doc, score in scored_candidates if score >= score_threshold
            ]
        else:
            filtered_candidates = [(doc, 1.0) for doc in candidates]

        # Извлечение уникальных Parent-документов
        seen_parents = set()
        parent_documents: List[Document] = []

        for child, score in filtered_candidates:
            p_id = child.metadata.get("parent_id")
            if not p_id or p_id in seen_parents:
                continue

            p_data = self.docstore.get(p_id)
            if p_data:
                seen_parents.add(p_id)
                meta = dict(p_data["metadata"])
                meta["rerank_score"] = round(score, 4)
                meta["matched_child_id"] = child.metadata.get("chunk_id")
                parent_documents.append(Document(page_content=p_data["page_content"], metadata=meta))

                if len(parent_documents) >= top_parents:
                    break

        return parent_documents, True

    def answer(
        self,
        query: str,
        top_parents: int = 3,
        score_threshold: float = 0.30,
        model_name: str = DEFAULT_LLM_MODEL,
        backend: str = "hf",
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.3
    ) -> str:
        """
        Полный RAG-пайплайн в один вызов:
        Поиск (FAISS + BM25) -> Реранкинг -> Извлечение Parent разделов -> Генерация через LLM.
        """
        docs, passed = self.search_and_rerank(
            query=query,
            top_parents=top_parents,
            score_threshold=score_threshold
        )
        if not passed or not docs:
            return "В базе данных Неолурка не найдено достаточно релевантной информации по вашему вопросу."
        return generate_answer(
            query=query,
            context_docs=docs,
            model_name=model_name,
            backend=backend,
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )


def build_hybrid_index(
    db_path: str = "neolurk.db",
    output_dir: str = DEFAULT_INDEX_DIR,
    limit: Optional[int] = None,
    model_name: str = DEFAULT_EMBED_MODEL,
    embeddings: Optional[Any] = None,
    batch_size: int = 512,
    save_interval: int = 1000,
    reset: bool = False
) -> HybridNeolurkRetriever:
    """
    Шаг 1: Построение Dense Vector Store (чистый FAISS Flat) + docstore.json с чекпоинтами.
    Чанки добавляются в FAISS батчами — потребление RAM стабильно ~1.5 ГБ (без OOM!).
    """
    t0 = time.time()
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")
    docstore_file = os.path.join(output_dir, "docstore.json")

    if embeddings is None:
        embeddings = get_embeddings(model_name, batch_size=batch_size)
    vectorstore: Optional[FAISS] = None
    docstore: Dict[str, Dict[str, Any]] = {}
    start_id = 0
    total_articles_indexed = 0

    # Проверка существующего чекпоинта
    if not reset and os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r", encoding="utf-8") as f:
                chk = json.load(f)
            start_id = chk.get("last_page_id", 0)
            total_articles_indexed = chk.get("articles_processed", 0)
            print(f"[+] ОБНАРУЖЕН ЧЕКПОИНТ: продолжение с ID статьи {start_id}")
            print(f"    - Уже обработано статей: {total_articles_indexed}")

            if os.path.exists(os.path.join(output_dir, "index.faiss")):
                vectorstore = FAISS.load_local(output_dir, embeddings, allow_dangerous_deserialization=True)
                print(f"    - Загружено векторов:    {vectorstore.index.ntotal}")

            if os.path.exists(docstore_file):
                with open(docstore_file, "r", encoding="utf-8") as f:
                    docstore = json.load(f)
                print(f"    - Загружено разделов:    {len(docstore)}")
        except Exception as e:
            print(f"[!] Ошибка чтения чекпоинта ({e}), начинаем сборку заново.")
            start_id = 0
            total_articles_indexed = 0
            vectorstore = None
            docstore = {}
    elif reset:
        print(f"[*] Флаг --reset: сборка индекса с нуля...")

    print(f"[*] Открытие базы: {db_path} (Read-Only)")
    archive = NeolurkArchive(db_path)
    chunker = NeolurkHierarchicalChunker()

    session_articles = 0
    batch: List[Document] = []
    current_last_id = start_id

    def commit_batch():
        """Сбрасывает накопленный батч чанков в FAISS."""
        nonlocal vectorstore
        if not batch:
            return
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.add_documents(batch)
        batch.clear()

    def save_checkpoint(last_id: int, is_final: bool = False):
        """Сохраняет текущий прогресс на диск с атомарной защитой от повреждения."""
        commit_batch()
        if vectorstore is not None:
            tmp_save_dir = os.path.join(output_dir, "_tmp_save")
            vectorstore.save_local(tmp_save_dir)
            for fname in os.listdir(tmp_save_dir):
                src = os.path.join(tmp_save_dir, fname)
                dst = os.path.join(output_dir, fname)
                os.replace(src, dst)
            if os.path.exists(tmp_save_dir):
                try:
                    os.rmdir(tmp_save_dir)
                except OSError:
                    pass

        tmp_docstore = docstore_file + ".tmp"
        with open(tmp_docstore, "w", encoding="utf-8") as f:
            json.dump(docstore, f, ensure_ascii=False)
        os.replace(tmp_docstore, docstore_file)

        chk_data = {
            "last_page_id": last_id,
            "articles_processed": total_articles_indexed + session_articles,
            "parents_count": len(docstore),
            "faiss_total": vectorstore.index.ntotal if vectorstore else 0,
            "model_name": model_name,
            "status": "completed" if is_final else "in_progress",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        tmp_chk = checkpoint_file + ".tmp"
        with open(tmp_chk, "w", encoding="utf-8") as f:
            json.dump(chk_data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_chk, checkpoint_file)


        gc.collect()
        total_st = total_articles_indexed + session_articles
        vs_cnt = vectorstore.index.ntotal if vectorstore else 0
        print(f"\n[💾 Чекпоинт сохранён] Статья ID: {last_id} | Статей: {total_st} | Разделов: {len(docstore)} | Векторов FAISS: {vs_cnt}")

    limit_str = f"{limit} статей" if limit else "до конца базы"
    print(f"[*] Обработка статей (лимит в этой сессии: {limit_str})...")

    try:
        for article in archive.iter_articles(start_id=start_id, limit=limit, clean_markdown=True):
            session_articles += 1
            current_last_id = article.id
            parents, children = chunker.chunk_article(article)

            # 1. Сохраняем родительские разделы в docstore
            for p in parents:
                docstore[p.metadata["parent_id"]] = {
                    "page_content": p.page_content,
                    "metadata": p.metadata
                }

            # 2. Накапливаем чанки в батч
            batch.extend(children)

            # 3. Как только набрался батч — сразу отправляем в FAISS
            if len(batch) >= batch_size:
                commit_batch()

            # 4. Прогресс-бар
            if session_articles % 25 == 0:
                faiss_cnt = vectorstore.index.ntotal if vectorstore else 0
                sys.stdout.write(
                    f"\r[+] Сессия: {session_articles} ст. | Всего ст.: {total_articles_indexed + session_articles} "
                    f"| Разделов: {len(docstore)} | Векторов: {faiss_cnt}"
                )
                sys.stdout.flush()

            # 5. Периодическое сохранение чекпоинта
            if session_articles % save_interval == 0:
                save_checkpoint(current_last_id, is_final=False)

    except KeyboardInterrupt:
        print("\n[!] Остановка пользователем. Сохраняем аварийный чекпоинт...")
        save_checkpoint(current_last_id, is_final=False)
        print("[+] Чекпоинт сохранён! Индекс готов к работе или продолжению сборки.")
        bm25 = None
        bm25s_dir = os.path.join(output_dir, "bm25s")
        bm25_file = os.path.join(output_dir, "bm25.pkl")
        if bm25s is not None and os.path.exists(bm25s_dir):
            try:
                bm25 = BM25SIndex.load(bm25s_dir)
            except Exception:
                pass
        elif os.path.exists(bm25_file):
            try:
                with open(bm25_file, "rb") as f:
                    bm25 = pickle.load(f)
            except Exception:
                pass
        return HybridNeolurkRetriever(vectorstore=vectorstore, bm25=bm25, docstore=docstore, model_name=model_name)

    # Финальное сохранение сессии
    save_checkpoint(current_last_id, is_final=True)
    print(f"\n[✔] Векторная база FAISS успешно построена за {time.time() - t0:.1f}с.")
    print(f"    - Обработано статей в сессии: {session_articles}")
    print(f"    - Всего статей в базе:        {total_articles_indexed + session_articles}")
    print(f"    - Всего разделов (Parent):    {len(docstore)}")
    print(f"    - Всего векторов в FAISS:     {vectorstore.index.ntotal}")

    bm25 = None
    bm25s_dir = os.path.join(output_dir, "bm25s")
    bm25_file = os.path.join(output_dir, "bm25.pkl")
    if bm25s is not None and os.path.exists(bm25s_dir):
        try:
            bm25 = BM25SIndex.load(bm25s_dir)
        except Exception:
            pass
    elif os.path.exists(bm25_file):
        try:
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)
        except Exception:
            pass

    return HybridNeolurkRetriever(vectorstore=vectorstore, bm25=bm25, docstore=docstore, model_name=model_name)


def build_bm25_index(
    output_dir: str = DEFAULT_INDEX_DIR,
    db_path: str = "neolurk.db",
    source: str = "parent"
) -> Any:
    """
    Шаг 2: Построение сверхбыстрого индекса BM25.
    Использует библиотеку bm25s (на C/NumPy) со стеммингом для русского языка.
    Запрос выполняется за 2-5 мс, потребление RAM в 4 раза меньше.
    """
    t0 = time.time()
    bm25s_dir = os.path.join(output_dir, "bm25s")
    bm25_file = os.path.join(output_dir, "bm25.pkl")

    if source == "parent":
        docstore_file = os.path.join(output_dir, "docstore.json")
        if not os.path.exists(docstore_file):
            raise FileNotFoundError(f"Файл {docstore_file} не найден. Сначала выполните build_hybrid_index().")
        print(f"[*] Загрузка разделов из {docstore_file} для BM25...")
        docstore = load_docstore(docstore_file)

        print(f"[*] Подготовка {len(docstore)} документов для BM25...")

        corpus = []
        for p_id, p_data in docstore.items():
            meta = dict(p_data["metadata"])
            meta["parent_id"] = p_id
            meta["chunk_id"] = p_id
            corpus.append({
                "page_content": p_data["page_content"],
                "metadata": meta
            })
    else:
        print(f"[*] Потоковое чтение статей из {db_path} для child-уровня BM25...")
        archive = NeolurkArchive(db_path)
        chunker = NeolurkHierarchicalChunker()
        corpus = []
        for art in archive.iter_articles(clean_markdown=True):
            _, children = chunker.chunk_article(art)
            for c in children:
                corpus.append({
                    "page_content": c.page_content,
                    "metadata": {"chunk_id": c.metadata.get("chunk_id"), "parent_id": c.metadata.get("parent_id")}
                })
            if len(corpus) % 100000 == 0:
                print(f"[+] Собрано чанков: {len(corpus)}...")

    # Если bm25s установлен — строим C/NumPy индекс
    if bm25s is not None:
        texts = [item["page_content"] for item in corpus]
        stemmer_fn = get_russian_stemmer()
        print(f"[*] Токенизация и стемминг {len(texts)} текстов через bm25s (русский язык)...")
        tokens = bm25s.tokenize(texts, stopwords="ru", stemmer=stemmer_fn)

        print("[*] Построение разреженной матрицы BM25S...")
        retriever = bm25s.BM25()
        retriever.index(tokens)

        print(f"[*] Сохранение индекса BM25S в '{bm25s_dir}'...")
        retriever.save(bm25s_dir, corpus=corpus)
        print(f"[✔] Сверхбыстрый индекс BM25S построен и сохранён в '{bm25s_dir}' за {time.time() - t0:.1f}с!")
        return BM25SIndex(retriever=retriever, corpus=corpus)
    else:
        print("[!] Пакет 'bm25s' не найден, используем классический rank_bm25...")
        bm25_docs = [Document(page_content=c["page_content"], metadata=c["metadata"]) for c in corpus]
        bm25 = BM25Retriever.from_documents(bm25_docs, preprocess_func=tokenize_ru)
        with open(bm25_file, "wb") as f:
            pickle.dump(bm25, f)
        print(f"[✔] Индекс BM25 сохранён в '{bm25_file}' за {time.time() - t0:.1f}с!")
        return bm25


class HFGenerator:
    """
    Генератор ответов на базе Hugging Face Transformers.
    Автоматически выбирает свободную GPU (например, cuda:1 при наличии 2 GPU)
    и кэширует модель в памяти.
    """
    def __init__(self, model_name: str = DEFAULT_LLM_MODEL, device: Optional[str] = None):
        import torch
        from transformers import pipeline

        if device is None:
            if torch.cuda.is_available():
                # Если доступно 2 GPU (как в Kaggle), используем cuda:1 для генерации,
                # оставляя cuda:0 для FAISS и эмбеддингов
                device = "cuda:1" if torch.cuda.device_count() > 1 else "cuda:0"
            else:
                device = "cpu"

        dtype = torch.float16 if "cuda" in str(device) else torch.float32

        print(f"[*] Загрузка LLM генератора: {model_name} на {device} ({dtype})...")
        self.device = device
        self.model_name = model_name
        self.pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=dtype,
            device=device,
            trust_remote_code=True
        )

    def generate(
        self,
        query: str,
        context_docs: List[Document],
        max_new_tokens: int = 512,
        temperature: float = 0.3
    ) -> str:
        if not context_docs:
            return "В базе данных Неолурка не найдено достаточно информации для ответа на этот вопрос."

        context_text = "\n\n---\n\n".join([
            f"Статья: {d.metadata.get('title')} | Раздел: {d.metadata.get('section')}\n"
            f"Синонимы: {d.metadata.get('synonyms_str', '')}\n"
            f"{d.page_content}"
            for d in context_docs
        ])

        system_prompt = (
            "Ты — эрудированный ассистент по энциклопедии интернет-культуры Неолурк.\n"
            "Ответь на вопрос пользователя, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленный ниже контекст.\n"
            "Если в контексте нет прямого ответа или информации недостаточно, прямо и честно скажи:\n"
            "'В базе Неолурка нет информации по данному вопросу'.\n"
            "Не выдумывай факты. Сохраняй стиль оригинала, если уместно."
        )

        user_content = f"Контекст из базы Неолурка:\n{context_text}\n\nВопрос: {query}\n\nОтвет:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

        tokenizer = self.pipe.tokenizer
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        outputs = self.pipe(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=(temperature > 0.0),
            pad_token_id=tokenizer.eos_token_id
        )

        generated_text = outputs[0]["generated_text"]
        if generated_text.startswith(prompt):
            answer = generated_text[len(prompt):].strip()
        else:
            answer = generated_text.strip()

        return answer


_cached_hf_generator: Optional[HFGenerator] = None


def get_hf_generator(model_name: str = DEFAULT_LLM_MODEL, device: Optional[str] = None) -> HFGenerator:
    global _cached_hf_generator
    if _cached_hf_generator is None or _cached_hf_generator.model_name != model_name:
        _cached_hf_generator = HFGenerator(model_name=model_name, device=device)
    return _cached_hf_generator


def generate_answer(
    query: str,
    context_docs: List[Document],
    model_name: str = DEFAULT_LLM_MODEL,
    backend: str = "hf",
    device: Optional[str] = None,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    base_url: str = "http://localhost:11434"
) -> str:
    """
    Универсальная генерация ответа на основе найденного контекста:
    - backend='hf' (по умолчанию): напрямую через Hugging Face Transformers (Qwen 2.5 3B / 7B).
      Не требует запущенных сторонних демонов или серверов.
    - backend='ollama': через локальный сервис Ollama.
    """
    if not context_docs:
        return "В базе данных Неолурка не найдено достаточно информации для ответа на этот вопрос."

    if backend == "hf":
        try:
            gen = get_hf_generator(model_name=model_name, device=device)
            return gen.generate(
                query=query,
                context_docs=context_docs,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
        except Exception as e:
            return f"[!] Ошибка генерации Hugging Face ({e})"

    elif backend == "ollama":
        if ChatOllama is None:
            return "[!] Пакет langchain-ollama не установлен. Установите: pip install langchain-ollama"

        context_text = "\n\n---\n\n".join([
            f"Статья: {d.metadata.get('title')} | Раздел: {d.metadata.get('section')}\n"
            f"Синонимы: {d.metadata.get('synonyms_str', '')}\n"
            f"{d.page_content}"
            for d in context_docs
        ])

        system_prompt = (
            "Ты — эрудированный ассистент по энциклопедии интернет-культуры Неолурк.\n"
            "Ответь на вопрос пользователя, опираясь ИСКЛЮЧИТЕЛЬНО на предоставленный ниже контекст.\n"
            "Если в контексте нет прямого ответа или информации недостаточно, прямо и честно скажи:\n"
            "'В базе Неолурка нет информации по данному вопросу'.\n"
            "Не выдумывай факты. Сохраняй стиль оригинала, если уместно."
        )

        user_prompt = f"Контекст из базы Неолурка:\n{context_text}\n\nВопрос: {query}\n\nОтвет:"

        try:
            llm = ChatOllama(model=model_name, base_url=base_url, temperature=temperature)
            response = llm.invoke([("system", system_prompt), ("user", user_prompt)])
            return response.content
        except Exception as e:
            return (
                f"[!] Ошибка подключения к Ollama ({e}).\n"
                f"Убедитесь, что Ollama запущена: `ollama run {model_name}`\n\n"
                f"Найденный контекст (без генерации):\n{context_text[:1000]}..."
            )
    else:
        raise ValueError(f"Неизвестный бэкенд: {backend}. Доступно: 'hf', 'ollama'")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hybrid RAG + Reranker + Ollama Qwen для Неолурка")
    parser.add_argument("--db-path", type=str, default="neolurk.db", help="Путь к файлу базы данных SQLite (neolurk.db)")
    parser.add_argument("--build", action="store_true", help="Собрать или продолжить сборку векторного индекса (Шаг 1)")
    parser.add_argument("--build-bm25", action="store_true", help="Собрать индекс BM25 (Шаг 2)")
    parser.add_argument("--bm25-source", type=str, default="parent", choices=["parent", "child"], help="Источник данных для BM25: 'parent' (по разделам) или 'child' (по чанкам)")
    parser.add_argument("--reset", action="store_true", help="Сбросить существующий чекпоинт и начать с нуля")
    parser.add_argument("--save-interval", type=int, default=1000, help="Интервал сохранения чекпоинта в статьях (по умолчанию 1000)")
    parser.add_argument("--batch-size", type=int, default=512, help="Размер батча векторизации FAISS")
    parser.add_argument("--limit", type=int, default=None, help="Лимит статей для обработки в этой сессии (по умолчанию вся база)")
    parser.add_argument("--query", type=str, default="Что такое МКАД?", help="Вопрос")
    parser.add_argument("--index-dir", type=str, default=DEFAULT_INDEX_DIR, help="Папка индекса")
    parser.add_argument("--threshold", type=float, default=0.30, help="Минимальный порог скора реранкера (0.0 - 1.0)")
    parser.add_argument("--no-rerank", action="store_true", help="Отключить реранкер")
    parser.add_argument("--reranker-model", type=str, default=DEFAULT_RERANKER_MODEL, help="Модель реранкера")
    parser.add_argument("--generate", action="store_true", help="Сгенерировать ответ через LLM")
    parser.add_argument("--backend", type=str, default="hf", choices=["hf", "ollama"], help="Бэкенд генерации: 'hf' (Hugging Face) или 'ollama'")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_LLM_MODEL, help="Модель Hugging Face (Qwen/Qwen2.5-3B-Instruct)")
    parser.add_argument("--ollama-model", type=str, default=DEFAULT_OLLAMA_MODEL, help="Модель Ollama (qwen2.5:7b)")
    
    args = parser.parse_args()

    # Шаг 2: Построение BM25
    if args.build_bm25:
        build_bm25_index(output_dir=args.index_dir, db_path=args.db_path, source=args.bm25_source)
        sys.exit(0)

    # Шаг 1: Построение или загрузка
    if args.build or not os.path.exists(os.path.join(args.index_dir, "checkpoint.json")):
        retriever = build_hybrid_index(
            db_path=args.db_path,
            output_dir=args.index_dir,
            limit=args.limit,
            batch_size=args.batch_size,
            save_interval=args.save_interval,
            reset=args.reset
        )
    else:
        print(f"[*] Загрузка индекса из '{args.index_dir}'...")
        retriever = HybridNeolurkRetriever.load(folder_path=args.index_dir)

    print("\n" + "=" * 70)
    print(f"ВОПРОС: «{args.query}»")
    print("=" * 70)

    # Поиск с реранкингом и порогом скора
    parents, passed = retriever.search_and_rerank(
        query=args.query,
        top_parents=2,
        k_candidates=15,
        score_threshold=args.threshold,
        rerank=(not args.no_rerank),
        reranker_model=args.reranker_model
    )

    if not passed:
        print(f"\n[!] ПОРОГ РЕЛЕВАНТНОСТИ НЕ ПРОЙДЕН (score < {args.threshold})")
        print("Ответ системы: «В базе данных Неолурка не найдено информации по этому вопросу.»")
    else:
        print(f"\n[+] Найдено релевантных разделов: {len(parents)}")
        for idx, p in enumerate(parents, 1):
            score_info = f"Score реранкера: {p.metadata.get('rerank_score')}" if not args.no_rerank else "Без реранкера"
            print(f"\n[{idx}] {score_info}")
            print(f"Статья:    {p.metadata.get('title')}")
            print(f"Раздел:    {p.metadata.get('section')}")
            print(f"Синонимы:  {p.metadata.get('synonyms_str')}")
            print(f"URL:       {p.metadata.get('url')}")
            print("-" * 70)
            print(p.page_content[:400].strip() + "\n...")

        if args.generate:
            model_info = args.llm_model if args.backend == "hf" else args.ollama_model
            print("\n" + "=" * 70)
            print(f"ГЕНЕРАЦИЯ ОТВЕТА ({args.backend.upper()}: {model_info}):")
            print("=" * 70)
            answer = generate_answer(
                args.query,
                parents,
                model_name=model_info,
                backend=args.backend
            )
            print(answer)
            print("=" * 70)

