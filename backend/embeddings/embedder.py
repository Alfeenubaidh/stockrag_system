"""
embedder.py — Embedding pipeline with local (SentenceTransformer) and remote (HF Inference API) modes.

Model: all-MiniLM-L6-v2 (384 dims)
Set USE_REMOTE_EMBEDDINGS=true to skip torch entirely and use HF Inference API.
"""

import logging
from typing import List, Dict
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------

@dataclass
class EmbeddingConfig:
    model_name: str = "all-MiniLM-L6-v2"
    batch_size: int = 128   # adjust to 64 if OOM on 4GB GPU
    max_chars: int = 2000
    normalize: bool = True


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

class EmbeddingPipeline:
    def __init__(self, config: EmbeddingConfig = EmbeddingConfig()):
        self.config = config

        from config.settings import settings

        self._remote = settings.use_remote_embeddings

        if self._remote:
            self._hf_api_key = settings.hf_api_key
            if not self._hf_api_key:
                raise ValueError("HF_API_KEY must be set when USE_REMOTE_EMBEDDINGS=true")
            self._hf_url = (
                f"https://api-inference.huggingface.co/models/"
                f"sentence-transformers/{self.config.model_name}"
            )
            self.dim = 384
            logger.info(f"Embedding pipeline in remote mode via HF Inference API: {self._hf_url}")
        else:
            import torch
            from sentence_transformers import SentenceTransformer

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = SentenceTransformer(self.config.model_name)
            self.model.to(self.device)

            logger.info(f"Embedding model running on: {self.device}")
            logger.info(f"Model device: {next(self.model.parameters()).device}")

            test_vec = self.model.encode(
                ["test"],
                convert_to_numpy=True,
                device=self.device
            )[0]

            self.dim = len(test_vec)

            if self.dim != 384:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected 384, got {self.dim}"
                )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _truncate(self, text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("Text must be string")
        return text[: self.config.max_chars]

    def _validate_texts(self, texts: List[str]) -> None:
        for i, t in enumerate(texts):
            if not t or not t.strip():
                raise ValueError(f"Empty text at index {i}")

    # ------------------------------------------------------------------
    # Core embedding
    # ------------------------------------------------------------------

    def _embed_batch_remote(self, texts: List[str]) -> List[List[float]]:
        import httpx

        response = httpx.post(
            self._hf_url,
            headers={"Authorization": f"Bearer {self._hf_api_key}"},
            json={"inputs": texts},
            timeout=30,
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"HF Inference API error {response.status_code}: {response.text}"
            )

        embeddings = response.json()

        if not isinstance(embeddings, list):
            raise TypeError(f"Unexpected HF API response type: {type(embeddings)}")

        if len(embeddings) != len(texts):
            raise ValueError(
                f"HF API returned {len(embeddings)} embeddings for {len(texts)} inputs"
            )

        return embeddings

    def _embed_batch_local(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=False,
            device=self.device,
        )

        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Model output is not numpy array")

        if embeddings.shape[1] != self.dim:
            raise ValueError(
                f"Embedding dimension mismatch: {embeddings.shape[1]} != {self.dim}"
            )

        return embeddings.tolist()

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        texts = [self._truncate(t) for t in texts]
        self._validate_texts(texts)

        if self._remote:
            return self._embed_batch_remote(texts)
        return self._embed_batch_local(texts)

    # ------------------------------------------------------------------
    # Public: embed raw texts
    # ------------------------------------------------------------------

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list):
            raise TypeError("texts must be a list")

        total = len(texts)
        results = []

        for i in range(0, total, self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            logger.info(f"Embedding batch {i}-{i+len(batch)-1} / {total}")

            embeddings = self._embed_batch(batch)
            results.extend(embeddings)

        logger.info(f"Embedded {total} texts total.")
        return results

    # ------------------------------------------------------------------
    # Public: embed chunks
    # ------------------------------------------------------------------

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        if not isinstance(chunks, list):
            raise TypeError("chunks must be a list")

        texts = []

        for i, c in enumerate(chunks):
            if "text" not in c:
                raise KeyError(f"Missing 'text' field in chunk index {i}")
            texts.append(c["text"])

        embeddings = self.embed_texts(texts)

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        return chunks

    # ------------------------------------------------------------------
    # Public: embed query
    # ------------------------------------------------------------------

    def embed_query(self, query: str) -> List[float]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be non-empty string")

        return self._embed_batch([query])[0]
