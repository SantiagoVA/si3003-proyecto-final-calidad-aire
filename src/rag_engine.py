"""
Motor de Recuperación Aumentada por Generación (RAG) con FAISS y Chunking Semántico.

Indexa el corpus normativo y técnico del Valle de Aburrá (`data/docs_rag/*.md`):
- Acuerdo Metropolitano N.° 04 de 2018 / 03 de 2019 (Protocolo POECA - AMVA)
- Resolución 2254 de 2017 (Ministerio de Ambiente - Norma Nacional e Índice ICA)
- Guía de Salud Pública, Grupos Sensibles y Actividad Física (SIATA / Secretaría de Salud)
- Dinámica Atmosférica, Topografía y Red de Monitoreo del Valle de Aburrá (SIATA / SISAIRE)

Combina embeddings densos multilingües (`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`)
en un índice vectorial `faiss.IndexFlatIP` con re-ranking léxico BM25 ligero para máxima precisión
tanto en consultas conceptuales como en artículos/umbrales exactos, retornando citas explícitas.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

try:
    import torch  # Importar antes de faiss evita conflictos libomp en macOS ARM64

    torch.set_num_threads(1)
except Exception:
    pass

import faiss
import numpy as np

faiss.omp_set_num_threads(1)

from src.data_loader import PROCESSED_DIR, PROJECT_ROOT

DOCS_RAG_DIR = PROJECT_ROOT / "data" / "docs_rag"
FAISS_INDEX_PATH = PROCESSED_DIR / "faiss_poeca.index"
CHUNKS_META_PATH = PROCESSED_DIR / "faiss_poeca_chunks.json"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def _normalize_tokens(text: str) -> List[str]:
    clean = "".join(
        c for c in unicodedata.normalize("NFD", text.lower())
        if unicodedata.category(c) != "Mn"
    )
    return [tok for tok in re.findall(r"[a-z0-9_.]+", clean) if len(tok) > 2]


def semantic_markdown_chunker(
    docs_dir: Path = DOCS_RAG_DIR,
    max_chars: int = 1150,
    overlap_chars: int = 160,
) -> List[Dict]:
    """
    Fragmenta documentos Markdown preservando jerarquía semántica de encabezados
    (`# Título Documento`, `## Artículo / Capítulo`, `### Sección`) y metadatos de citación.
    """
    chunks: List[Dict] = []
    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No se encontraron documentos normativos en {docs_dir}")

    for file_path in md_files:
        raw_text = file_path.read_text(encoding="utf-8")
        lines = raw_text.splitlines()

        doc_title = file_path.stem
        doc_code = file_path.stem.upper()
        h2_current = "Disposiciones Generales"
        h3_current = ""
        buffer: List[str] = []

        def _flush_buffer():
            nonlocal buffer
            body = "\n".join(buffer).strip()
            buffer = []
            if len(body) < 40:
                return
            section_label = f"{h2_current} — {h3_current}" if h3_current else h2_current
            # Si una sección excede max_chars, subdividir por párrafos con solapamiento
            if len(body) <= max_chars:
                sub_texts = [body]
            else:
                paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
                sub_texts = []
                cur = ""
                for p in paragraphs:
                    if len(cur) + len(p) + 2 <= max_chars:
                        cur = f"{cur}\n\n{p}".strip()
                    else:
                        if cur:
                            sub_texts.append(cur)
                        tail = cur[-overlap_chars:] if cur else ""
                        cur = f"{tail}\n{p}".strip() if tail else p
                if cur:
                    sub_texts.append(cur)

            for part_idx, sub in enumerate(sub_texts):
                sec_full = section_label if len(sub_texts) == 1 else f"{section_label} (Parte {part_idx + 1})"
                citation = f"[Fuente: {doc_title} ({doc_code}), {sec_full}]"
                enriched_text = f"Documento: {doc_title}\nSección: {sec_full}\nContenido:\n{sub}"
                chunks.append(
                    {
                        "chunk_id": f"{doc_code}_{len(chunks):03d}",
                        "archivo": file_path.name,
                        "doc_code": doc_code,
                        "documento": doc_title,
                        "seccion": sec_full,
                        "contenido": sub,
                        "texto_indexado": enriched_text,
                        "cita": citation,
                    }
                )

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("# ") and not stripped.startswith("## "):
                doc_title = stripped.lstrip("# ").strip()
            elif "**Código Documento:**" in stripped:
                doc_code = stripped.split("**Código Documento:**")[-1].strip()
            elif stripped.startswith("## "):
                _flush_buffer()
                h2_current = stripped.lstrip("# ").strip()
                h3_current = ""
            elif stripped.startswith("### "):
                _flush_buffer()
                h3_current = stripped.lstrip("# ").strip()
            elif stripped == "---":
                continue
            else:
                buffer.append(line)

        _flush_buffer()

    return chunks


class POECARagEngine:
    """
    Motor RAG local con FAISS (`IndexFlatIP` sobre vectores L2-normalizados = similitud coseno)
    y re-ranking híbrido léxico-semántico para consulta normativa y de salud pública.
    """

    def __init__(
        self,
        docs_dir: Path = DOCS_RAG_DIR,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.docs_dir = docs_dir
        self.embedding_model_name = embedding_model_name
        self.chunks: List[Dict] = []
        self.index: Optional[faiss.Index] = None
        self.backend_type: str = "sentence-transformers"
        self._embedder = None
        self._tfidf_vectorizer = None

    def _load_embedder(self):
        if self._embedder is not None or self._tfidf_vectorizer is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
            self.backend_type = "sentence-transformers"
        except Exception:
            # Plan B offline si no estuviera disponible el modelo neuronal en otro entorno
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=2048)
            self.backend_type = "tfidf-faiss-fallback"

    def _encode_texts(self, texts: List[str], fit_fallback: bool = False) -> np.ndarray:
        self._load_embedder()
        if self._embedder is not None:
            vecs = self._embedder.encode(
                texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
            )
            return np.asarray(vecs, dtype=np.float32)
        if fit_fallback:
            mat = self._tfidf_vectorizer.fit_transform(texts).toarray().astype(np.float32)
        else:
            mat = self._tfidf_vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms < 1e-8] = 1.0
        return mat / norms

    def build_or_load(self, force_rebuild: bool = False) -> "POECARagEngine":
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        if (
            not force_rebuild
            and FAISS_INDEX_PATH.exists()
            and CHUNKS_META_PATH.exists()
        ):
            meta = json.loads(CHUNKS_META_PATH.read_text(encoding="utf-8"))
            if meta.get("backend_type") == "sentence-transformers":
                self.chunks = meta["chunks"]
                self.backend_type = meta["backend_type"]
                self.index = faiss.read_index(str(FAISS_INDEX_PATH))
                return self

        self.chunks = semantic_markdown_chunker(self.docs_dir)
        texts = [c["texto_indexado"] for c in self.chunks]
        embeddings = self._encode_texts(texts, fit_fallback=True)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        if self.backend_type == "sentence-transformers":
            faiss.write_index(self.index, str(FAISS_INDEX_PATH))
            CHUNKS_META_PATH.write_text(
                json.dumps(
                    {
                        "embedding_model": self.embedding_model_name,
                        "backend_type": self.backend_type,
                        "num_chunks": len(self.chunks),
                        "dimension": int(dim),
                        "chunks": self.chunks,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        return self

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        if not query or not isinstance(query, str) or len(query.strip()) < 2:
            raise ValueError("La consulta normativa al motor RAG no puede estar vacía.")
        if self.index is None or not self.chunks:
            self.build_or_load()

        q_vec = self._encode_texts([query], fit_fallback=False)
        cand_k = min(len(self.chunks), max(top_k * 3, 8))
        scores, indices = self.index.search(q_vec, cand_k)

        q_tokens = set(_normalize_tokens(query))
        reranked = []
        for dense_score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[int(idx)]
            c_tokens = set(_normalize_tokens(chunk["texto_indexado"]))
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), 1)
            hybrid_score = 0.78 * float(dense_score) + 0.22 * overlap
            reranked.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "documento": chunk["documento"],
                    "doc_code": chunk["doc_code"],
                    "seccion": chunk["seccion"],
                    "contenido": chunk["contenido"],
                    "cita": chunk["cita"],
                    "score_semantico": round(float(dense_score), 4),
                    "score_hibrido": round(float(hybrid_score), 4),
                }
            )

        reranked.sort(key=lambda r: r["score_hibrido"], reverse=True)
        return reranked[:top_k]


_SINGLETON_RAG: Optional[POECARagEngine] = None


def get_rag_engine(force_rebuild: bool = False) -> POECARagEngine:
    global _SINGLETON_RAG
    if _SINGLETON_RAG is None or force_rebuild:
        _SINGLETON_RAG = POECARagEngine().build_or_load(force_rebuild=force_rebuild)
    return _SINGLETON_RAG


if __name__ == "__main__":
    engine = get_rag_engine(force_rebuild=True)
    print(f"Índice FAISS construido: {len(engine.chunks)} fragmentos | Backend: {engine.backend_type}")
    for r in engine.search("¿Qué restricciones aplican en Estado de Alerta Roja del POECA y para camiones?", top_k=2):
        print(" ->", r["score_hibrido"], r["cita"])
