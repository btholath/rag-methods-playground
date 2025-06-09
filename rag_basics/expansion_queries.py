import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
from utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import umap
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# ========= 1. Environment Setup =========
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# ========= 2. Load and Extract Text from PDFs =========
pdf_paths = [
    "../data/65223.pdf",  # Accumulator 10 Product Guide
    "../data/65175.pdf",  # Accumulator 10 Product Brochure
]
pdf_texts = []
for pdf_path in pdf_paths:
    reader = PdfReader(pdf_path)
    pdf_texts.extend([p.extract_text().strip() for p in reader.pages if p.extract_text()])

pdf_texts = [t for t in pdf_texts if t]

# ========= 3. Text Chunking =========
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# ========= 4. Embedding & Vector Store =========
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "athene-accumulator10-collection", embedding_function=embedding_function
)
ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

# ========= 5. Generate Related Queries with OpenAI =========
def generate_related_queries(query, model="gpt-3.5-turbo"):
    prompt = (
        "You are a helpful insurance assistant. "
        "Given the following question, propose up to five related questions "
        "that would help someone understand the Athene Accumulator 10 annuity product. "
        "Make each question concise, focused, and on a separate line."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    return [line.strip() for line in content.strip().split("\n") if line.strip()]

original_query = "What are the main features and surrender charges of Athene Accumulator 10?"
aug_queries = generate_related_queries(original_query)

# Show the augmented queries
print("\nRelated Queries:")
for q in aug_queries:
    print("-", q)

# ========= 6. Run All Queries and Retrieve Documents =========
joint_queries = [original_query] + aug_queries
results = chroma_collection.query(
    query_texts=joint_queries, n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"]

# Deduplicate the retrieved documents
unique_documents = set()
for docs in retrieved_documents:
    unique_documents.update(docs)

# Output the results
for i, docs in enumerate(retrieved_documents):
    print(f"\nQuery: {joint_queries[i]}\nResults:")
    for doc in docs:
        print(word_wrap(doc))
        print()
    print("-" * 80)

# ========= 7. Embedding Projection & Visualization =========
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

original_query_embedding = embedding_function([original_query])
augmented_query_embeddings = embedding_function(joint_queries)
project_original_query = project_embeddings(original_query_embedding, umap_transform)
project_augmented_queries = project_embeddings(augmented_query_embeddings, umap_transform)

retrieved_embeddings = results["embeddings"]
result_embeddings = [emb for sublist in retrieved_embeddings for emb in sublist]
projected_result_embeddings = project_embeddings(result_embeddings, umap_transform)

plt.figure(figsize=(8, 6))
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
    label="All Chunks"
)
plt.scatter(
    project_augmented_queries[:, 0],
    project_augmented_queries[:, 1],
    s=150,
    marker="X",
    color="orange",
    label="Queries"
)
plt.scatter(
    projected_result_embeddings[:, 0],
    projected_result_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
    label="Top Retrieved"
)
plt.scatter(
    project_original_query[:, 0],
    project_original_query[:, 1],
    s=150,
    marker="X",
    color="r",
    label="Original Query"
)
plt.gca().set_aspect("equal", "datalim")
plt.title("Athene Accumulator 10 - Query and Document Embedding Space")
plt.axis("off")
plt.legend()
# (after plotting as before)
plt.savefig("expansion_queries_plot.png", bbox_inches="tight")
print("Plot saved as expansion_queries_plot.png")
# plt.show()  # Omit this line in headless environment