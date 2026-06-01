"""e5 passage embedder backed by sentence-transformers (HuggingFace, no company deps).

`sentence-transformers`/`torch` are imported lazily inside ``__init__`` so this module
can be imported without them present.
"""

import os
# noinspection PyPackageRequirements
from transformers import PreTrainedTokenizerBase

from ..config import IndexingConfig
from .engines import HuggingFaceTokenizer


# noinspection PyNoneFunctionAssignment,PyArgumentList,PyUnresolvedReferences
class E5Embedder:
    """Loads `intfloat/multilingual-e5-small` once and embeds passages.

    The same HuggingFace tokenizer is exposed (via :attr:`tokenizer`) for the chunker,
    so token counting/alignment and embedding use one consistent tokenizer.
    """

    def __init__(self, config: IndexingConfig) -> None:
        if config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from sentence_transformers import SentenceTransformer  # lazy: pulls torch

        model_id = config.model_path or config.embedding_model
        self._model = SentenceTransformer(model_id, device=config.device or None)
        self._prefix = config.passage_prefix
        self._batch_size = config.embedding_batch_size
        self._hf_tokenizer: PreTrainedTokenizerBase = self._model.tokenizer
        self._tokenizer = HuggingFaceTokenizer(self._hf_tokenizer)

    @property
    def tokenizer(self) -> HuggingFaceTokenizer:
        """The shared tokenizer adapter (implements the chunker's `Tokenizer` protocol)."""
        return self._tokenizer

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        """Embed chunk texts with the ``passage:`` prefix and L2-normalization."""
        if not texts:
            return []
        inputs = [f"{self._prefix}{text}" for text in texts]
        vectors = self._model.encode(
            inputs,
            normalize_embeddings=True,
            batch_size=self._batch_size,
        )
        return [list(map(float, vector)) for vector in vectors]

    def count_tokens(self, text: str) -> int:
        """Full-document token count (no special tokens), pre-chunking."""
        return len(self._hf_tokenizer(text, add_special_tokens=False)["input_ids"])

    def max_content_tokens(self) -> int:
        """Usable content tokens per chunk = model max - special tokens - prefix tokens."""
        model_max = getattr(self._model, "max_seq_length", 512) or 512
        special = self._hf_tokenizer.num_special_tokens_to_add(pair=False)
        prefix_tokens = len(self._hf_tokenizer(self._prefix, add_special_tokens=False)["input_ids"])
        return max(1, model_max - special - prefix_tokens)
