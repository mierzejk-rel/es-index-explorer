"""e5 passage embedder backed by sentence-transformers (HuggingFace, no company deps).

`sentence-transformers`/`torch` are imported lazily inside ``__init__`` so this module
can be imported without them present.
"""

import os
# noinspection PyPackageRequirements
from transformers import PreTrainedTokenizerBase

from ..config import IndexingConfig
from .engines import (
    HuggingFaceTokenizer,
    silence_docopt_syntax_warnings,
    silence_transformers_alias_warnings,
)


# noinspection PyNoneFunctionAssignment,PyArgumentList,PyUnresolvedReferences
class E5Embedder:
    """Loads `intfloat/multilingual-e5-small` once and embeds passages.

    The same HuggingFace tokenizer is exposed (via :attr:`tokenizer`) for the chunker,
    so token counting/alignment and embedding use one consistent tokenizer.
    """

    def __init__(self, config: IndexingConfig) -> None:
        if config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        silence_transformers_alias_warnings()
        silence_docopt_syntax_warnings()
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


class SparseInputTooLongError(ValueError):
    """Raised when a metadata field exceeds the sparse encoder's token budget."""

    def __init__(self, field_name: str, token_count: int, token_budget: int) -> None:
        self.field_name = field_name
        self.token_count = token_count
        self.token_budget = token_budget
        super().__init__(
            f"'{field_name}' has {token_count} tokens, which exceeds sparse token budget {token_budget}."
        )


class SparseEmbedder:
    """Client-side sparse encoder for metadata fields."""

    def __init__(self, config: IndexingConfig) -> None:
        if config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        silence_transformers_alias_warnings()
        silence_docopt_syntax_warnings()
        from sentence_transformers.sparse_encoder import SparseEncoder  # lazy: pulls torch

        model_id = config.sparse_model_path or config.sparse_model
        device = config.sparse_device or config.device
        self._model = SparseEncoder(model_id, device=device or None)
        self._hf_tokenizer: PreTrainedTokenizerBase = self._model.tokenizer
        model_max = getattr(self._model, "max_seq_length", 512) or 512
        special = self._hf_tokenizer.num_special_tokens_to_add(pair=False)
        derived_budget = max(1, model_max - special)
        self._token_budget = config.sparse_max_tokens or derived_budget

    def token_budget(self) -> int:
        """Return max metadata tokens allowed by the sparse encoder."""
        return self._token_budget

    def count_tokens(self, text: str) -> int:
        """Count metadata field tokens without special tokens."""
        return len(self._hf_tokenizer(text, add_special_tokens=False)["input_ids"])

    def encode(self, text: str, *, field_name: str = "metadata") -> dict[str, float]:
        """Encode metadata text as sparse token weights for Elasticsearch."""
        token_count = self.count_tokens(text)
        budget = self.token_budget()
        if token_count > budget:
            raise SparseInputTooLongError(field_name, token_count, budget)

        encoded = self._model.encode(text, output_value="token_weights")
        if isinstance(encoded, list):
            if not encoded:
                return {}
            encoded = encoded[0]
        if not isinstance(encoded, dict):
            raise TypeError(f"Unexpected sparse encode output type: {type(encoded).__name__}")
        return {str(token): float(weight) for token, weight in encoded.items()}
