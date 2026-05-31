"""Document indexing pipeline: chunking, embedding, and Elasticsearch writing.

This package is intentionally light to import. Heavy optional dependencies
(`sentence-transformers`/`torch`, `wtpsplit`, `spacy`) are imported lazily inside
the relevant engines/adapters so that the pure chunking algorithm in
``es_index_explorer.indexing.chunking`` can be imported and unit-tested without them.
"""
