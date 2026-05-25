# Elasticsearch Index Report: 51b1e1f7-f607-4967-a439-cf9babafab2c-1030345-qna-subsetting

Generated: `2026-05-25T12:18:45.293433+00:00`

## Overview
- Documents: `242`
- Size (MB): `1.45`
- Shards: `1` primary / `1` replica

## Retrieval Hints
- Text fields (BM25): body, title
- Keyword fields (filters): controlNumber, subsetIds
- Vector fields (kNN): embedding

## Field Usage
- _id: doc_values=0, knn_vectors=0, norms=0, offsets=0, payloads=0, points=0, positions=0, postings=0, proximity=0, stored_fields=20*, term_frequencies=0, term_vectors=0, terms=1*
- _source: doc_values=0, knn_vectors=0, norms=0, offsets=0, payloads=0, points=0, positions=0, postings=0, proximity=0, stored_fields=20*, term_frequencies=0, term_vectors=0, terms=0
- body: doc_values=0, knn_vectors=0, norms=10*, offsets=0, payloads=0, points=0, positions=0, postings=10*, proximity=0, stored_fields=0, term_frequencies=10*, term_vectors=0, terms=10*
- chunkId: doc_values=10*, knn_vectors=0, norms=0, offsets=0, payloads=0, points=10*, positions=0, postings=0, proximity=0, stored_fields=0, term_frequencies=0, term_vectors=0, terms=0
- controlNumber: doc_values=1*, knn_vectors=0, norms=0, offsets=0, payloads=0, points=0, positions=0, postings=0, proximity=0, stored_fields=0, term_frequencies=0, term_vectors=0, terms=0
- documentId: doc_values=0, knn_vectors=0, norms=0, offsets=0, payloads=0, points=10*, positions=0, postings=0, proximity=0, stored_fields=0, term_frequencies=0, term_vectors=0, terms=0
- subsetIds: doc_values=0, knn_vectors=0, norms=0, offsets=0, payloads=0, points=0, positions=0, postings=11*, proximity=0, stored_fields=0, term_frequencies=0, term_vectors=0, terms=11*

## Analysis Settings
```json
{}
```

## Field Details
- body: type=text
- chunkId: type=integer
- chunkSize: type=integer
- controlNumber: type=keyword
- createdAt: type=date
- documentId: type=integer
- documentModifyTime: type=date
- embedding: type=dense_vector, index=True, dims=384, similarity=cosine, index_options={"type": "bbq_hnsw", "m": 16, "ef_construction": 100, "rescore_vector": {"oversample": 3.0}}
- subsetIds: type=keyword
- title: type=text