import pandas as pd
import csv
import json
import build_ground_truth_embeddings
from haystack import Document
from haystack.utils import Secret
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.retrievers import InMemoryEmbeddingRetriever, MultiQueryEmbeddingRetriever
from haystack.components.builders import ChatPromptBuilder
from haystack.dataclasses import ChatMessage

import os
from getpass import getpass
from haystack.components.generators.chat import HuggingFaceLocalChatGenerator
from haystack import Pipeline


def calculate_p_at_k(retrieved_ids, relevant_ids, k=10):
    """Calculate Precision at k"""
    retrieved_at_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    relevant_at_k = len(retrieved_at_k.intersection(relevant_set))
    return relevant_at_k / k


def evaluate_retrieval(retriever_result, ground_truth, query_ids, k=10):
    """
    Evaluate retrieval performance using P@k metric
    
    Args:
        retriever_result: Result from retriever.run(queries=queryList) - dict with 'documents' key
        ground_truth: Dict mapping query_id to list of relevant doc IDs
        query_ids: List of query IDs in same order as queryList
        k: Value of k for P@k metric (default 10 for P@10)
    
    Returns:
        Dict with evaluation metrics
    """
    p_at_k_scores = []
    
    # Get all retrieved documents
    all_docs = retriever_result.get('documents', [])
    
    # Assume documents are in order: top_k for first query, then next, etc.
    docs_per_query = len(query_ids)
    docs_per_query_count = len(all_docs) // docs_per_query if docs_per_query > 0 else 0
    
    for idx, query_id in enumerate(query_ids):
        if query_id in ground_truth:
            # Get documents for this query (slice based on index)
            start_idx = idx * docs_per_query_count
            end_idx = start_idx + docs_per_query_count
            retrieved_docs_for_query = all_docs[start_idx:end_idx]
            
            # Extract document IDs
            retrieved_ids = [str(doc.id) for doc in retrieved_docs_for_query]
            relevant_ids = [str(rid) for rid in ground_truth[query_id]]
            
            p_at_k = calculate_p_at_k(retrieved_ids, relevant_ids, k=k)
            p_at_k_scores.append(p_at_k)
    
    # Calculate mean P@k
    if p_at_k_scores:
        mean_p_at_k = sum(p_at_k_scores) / len(p_at_k_scores)
        return {
            f"P@{k}": mean_p_at_k,
            f"P@{k}_scores": p_at_k_scores,
            "num_queries_evaluated": len(p_at_k_scores)
        }
    else:
        return {f"P@{k}": 0, f"P@{k}_scores": [], "num_queries_evaluated": 0}


embeddDict = build_ground_truth_embeddings.getQueries_Docs()
document_store = embeddDict.get("docsStore")

# df = pd.read_csv('fulldocs.tsv', sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False)
# cols = [c.lower() for c in df.columns]
#     # prefer common text columns, otherwise use last column
# for candidate in ("text", "content", "body", "document", "doc"):
#     if candidate in cols:
#         text_col = df.columns[cols.index(candidate)]
#         break
# else:
#     text_col = df.columns[-1]

# docs = []
# for idx, row in df.iterrows():
#     content = str(row[text_col]).strip()
#     if not content:
#         continue
#     meta = {k: str(v) for k, v in row.items() if k != text_col}
#     docs.append(Document(content=content, meta=meta, id=str(idx)))

doc_embedder = SentenceTransformersDocumentEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")
doc_embedder.warm_up()

# queries = pd.read_csv('queries.tsv', sep="\t", quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, names=['id', 'query'])
querySamples = embeddDict.get("queries")
queryList=querySamples['query'].tolist()
query_ids = querySamples['id'].tolist()
print(f"Query list: {queryList}")
print(f"Query IDs: {query_ids}")

# docs_with_embeddings = doc_embedder.run(docs)
# document_store.write_documents(docs_with_embeddings["documents"])

text_embedder = SentenceTransformersTextEmbedder(model="sentence-transformers/all-MiniLM-L6-v2")

retriever = MultiQueryEmbeddingRetriever(
    retriever=InMemoryEmbeddingRetriever(document_store=document_store, top_k=10),
    query_embedder=text_embedder
)

#retriever = InMemoryEmbeddingRetriever(document_store)

template = [
    ChatMessage.from_user(
        """
Given the following information, answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{question}}
Answer:
"""
    )
]

prompt_builder = ChatPromptBuilder(template=template)


chat_generator = HuggingFaceLocalChatGenerator(model="Qwen/Qwen3-4B-Instruct-2507")


basic_rag_pipeline = Pipeline()

# Add components to pipeline
basic_rag_pipeline.add_component("retriever", retriever)
basic_rag_pipeline.add_component("prompt_builder", prompt_builder)
basic_rag_pipeline.add_component("llm", chat_generator)

# Now, connect the components to each other
basic_rag_pipeline.connect("retriever.documents", "prompt_builder.documents")
basic_rag_pipeline.connect("prompt_builder.prompt", "llm.messages")

# question = "What is Paris, France like?"

# Run the components step by step to access retrieved documents
retriever_result = retriever.run(queries=queryList)

# ===== P@10 Evaluation =====
# Load ground truth if available
try:
    with open('ground_truth.json', 'r') as f:
        ground_truth = json.load(f)
    
    # Evaluate retrieval performance
    metrics = evaluate_retrieval(retriever_result, ground_truth, query_ids, k=10)
    
    print("\n" + "="*50)
    print("EVALUATION METRICS - P@10 (Precision at 10)")
    print("="*50)
    print(f"Mean P@10: {metrics['P@10']:.4f}")
    print(f"Queries evaluated: {metrics['num_queries_evaluated']}")
    print(f"Individual P@10 scores: {[f'{score:.4f}' for score in metrics['P@10_scores']]}")
    print("="*50 + "\n")
except FileNotFoundError:
    print("Note: ground_truth.json not found. Run build_ground_truth.py to generate it.")
except Exception as e:
    print(f"Evaluation error: {e}")

prompt_result=[]
llm_result=[]
for q in querySamples:
    # Use queries=[q] for consistent retrieval API
    retrieved_docs = retriever_result["documents"]
    prompt_result=prompt_builder.run(documents=retrieved_docs, question=q)
    llm_result=chat_generator.run(messages=prompt_result["prompt"])
    print("\nAnswer:",llm_result["results"][0].text)

# for p in prompt_result: 
#     llm_result.append(chat_generator.run(messages=p["prompt"]))

# for r in llm_result:
#     print("\nAnswer:", r["replies"][0].text)
    
print("\nTop 10 documents used:")
for i, doc in enumerate(retrieved_docs, 1):
    print(f"{i}. {doc.meta}")