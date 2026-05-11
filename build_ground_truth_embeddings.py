import pandas as pd
import json
import csv
import numpy as np
from haystack import Document
from pathlib import Path
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder, SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever, MultiQueryEmbeddingRetriever

def getQueries_Docs():
    # Load documents
    fileDocuments = Path("documents.json")

    if not fileDocuments.is_file():
        print("Loading documents...")
        df = pd.read_csv('fulldocs.tsv', sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
        cols = [c.lower() for c in df.columns]

        # Find text column
        for candidate in ("text", "content", "body", "document", "doc"):
            if candidate in cols:
                text_col = df.columns[cols.index(candidate)]
                break
        else:
            text_col = df.columns[-1]

        docs = []
        for idx, row in df.iterrows():
            content = str(row[text_col]).strip()
            if not content:
                continue
            meta = {k: str(v) for k, v in row.items() if k != text_col}
            docs.append(Document(content=content, meta=meta, id=str(idx)))

        print(f"Loaded {len(docs)} documents")

        # Embed documents
        print("Embedding documents...")
        doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
        doc_embedder.warm_up()
        docs_with_embeddings = doc_embedder.run(docs)

        # Create document store
        document_store = InMemoryDocumentStore()
        document_store.write_documents(docs_with_embeddings["documents"])
        document_store.save_to_disk("documents.json")
    else:
        document_store = InMemoryDocumentStore()
        document_store.load_from_disk("documents.json")


    # Load queries
    print("Loading queries...")
    queries_df = pd.read_csv('queries.tsv', sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, names=['id', 'query'])
    queries_rand = queries_df.sample(n=10)

    # Embedder for queries
    text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

    # Generate ground truth
    print("Generating ground truth using embedding similarity...")
    ground_truth = {}

    retriever = MultiQueryEmbeddingRetriever(
        retriever=InMemoryEmbeddingRetriever(document_store=document_store, top_k=20),
        query_embedder=text_embedder
    )

    for idx, row in queries_rand.iterrows():
        query_id = str(row['id'])
        query_text = row['query']
        
        # Retrieve top 20 similar documents
        result = retriever.run(queries=[query_text])
        relevant_doc_ids = [str(doc.id) for doc in result['documents']]
        
        ground_truth[query_id] = relevant_doc_ids
        
        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1}/{len(queries_df)} queries...")

    print(f"Generated ground truth for {len(ground_truth)} queries")

    # Save ground truth
    with open('ground_truth.json', 'w') as f:
        json.dump(ground_truth, f, indent=2)

    print("Ground truth saved to ground_truth.json")
    print(f"\nExample: Query {list(ground_truth.keys())[0]} has {len(list(ground_truth.values())[0])} relevant documents")

    embeddings = {"queries":queries_rand,"docsStore":document_store}
    return embeddings