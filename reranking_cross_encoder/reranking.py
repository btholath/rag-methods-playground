import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import numpy as np
from sentence_transformers import CrossEncoder
from openai import OpenAI

# Utility for pretty-print
def word_wrap(text, width=90):
    import textwrap
    return '\n'.join(textwrap.wrap(text, width=width))

# ========== 1. Load environment and clients ==========
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

embedding_function = SentenceTransformerEmbeddingFunction()

# ========== 2. Load & Extract Text from PDFs ==========
pdf_paths = [
    "./data/65223.pdf",  # Accumulator 10 Product Guide
    "./data/65175.pdf",  # Accumulator 10 Product Brochure
]

all_texts = []
for path in pdf_paths:
    reader = PdfReader(path)
    texts = [p.extract_text().strip() for p in reader.pages if p.extract_text()]
    all_texts.extend(texts)

# ========== 3. Chunking ==========
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(all_texts))

token_splitter = SentenceTransformersTokenTextSplitter(
    chunk_overlap=0, tokens_per_chunk=256
)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# ========== 4. Embedding & Chroma Vector DB ==========
chroma_client = chromadb.Client()
chroma_collection = chroma_client.get_or_create_collection(
    "accumulator10-collect", embedding_function=embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
print(f"Added {len(token_split_texts)} chunks to vector DB.")

# ========== 5. Queries (Original & Generated) ==========
original_query = "What are the surrender charges and withdrawal options for Athene Accumulator 10?"
generated_queries = [
    "What is the minimum and maximum premium allowed?",
    "Explain the free withdrawal provisions.",
    "Are there any death benefits in this annuity?",
    "How does the terminal illness waiver work?",
    "What market indices are used for interest crediting?",
    "What is the confinement waiver and who is eligible?",
    "Are there preset allocation options for investment strategies?",
    "How is the bailout feature triggered?",
    "Is this product available in California?",
    "What are the issue ages for Accumulator 10?",
]

all_queries = [original_query] + generated_queries

# ========== 6. Retrieve Relevant Documents ==========
results = chroma_collection.query(
    query_texts=all_queries, n_results=10, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Deduplicate retrieved docs
unique_documents = set()
for documents in retrieved_documents:
    for document in documents:
        unique_documents.add(document)
unique_documents = list(unique_documents)

# ========== 7. Rerank with CrossEncoder ==========
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
pairs = [[original_query, doc] for doc in unique_documents]
scores = cross_encoder.predict(pairs)

print("\nTop 5 most relevant docs (reranked):\n")
top_indices = np.argsort(scores)[::-1][:5]
top_documents = [unique_documents[i] for i in top_indices]
for i, doc in enumerate(top_documents, 1):
    print(f"Document #{i} (Score: {scores[top_indices[i-1]]:.4f})")
    print(word_wrap(doc))
    print("=" * 100)

# ========== 8. Generate Final Answer using OpenAI ==========
def generate_multi_query(query, context, model="gpt-3.5-turbo"):
    prompt = """
    You are a knowledgeable financial research assistant. 
    Your users are inquiring about annuity products from Athene. 
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Based on the following context:\n\n{context}\n\nAnswer the query: '{query}'"},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    return content

context = "\n\n".join(top_documents)
res = generate_multi_query(query=original_query, context=context)
print("\nFinal Answer:\n")
print(res)
