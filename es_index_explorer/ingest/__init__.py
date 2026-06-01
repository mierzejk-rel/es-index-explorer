"""Unified RelativityOne -> Elasticsearch ingest engine.

This package converges the two former entrypoints (export-based reading and
QuerySlim-based importing) into one engine driven by a pluggable `DocumentSource`.
The engine, progress logging, and error reporting are identical regardless of the
read mechanism; only how a batch of documents is fetched differs.

Heavy ML dependencies are imported lazily by the indexing pipeline, so importing
this package (or running a dry-run read) requires neither torch nor transformers.
"""
