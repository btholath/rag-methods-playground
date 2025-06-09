"""
Query Expansion (with generated answers)
Generate potential answers to the query [using an LLM] and to get relevant context.

Conceptual Flow (From enhanced query formation to post-retrieval refinement):
Original Query -> LLM -> **Answer (Contextual Add-on) -> Vector DB -> Query Results -> LLM -> Answer
The Origianl Query and the Hallucinated **Answer -> Concatenate & Form Augmented Query -> Query Vector Database to Retrieve Relevant Docs ->
-> Process Retrieved Docs through LLM (Re-ranking) -> Final, Context-Enhanced Answer/Result


Use Cases & Applications:
- Search Engines: Improved query expansion leads to more comprehensive search results.

- Question Answering Systems: More relevant documents or passages are retrieved for better answer synthesis.

- E-Commerce Product Search: Better matching of user intent with product metadata improves accuracy.

- Academic Research: Expanding search queries with related scientific terms helps in fetching a broader range of relevant literature.

"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="n_jobs value 1 overridden")
from utils import project_embeddings, word_wrap
from pypdf import PdfReader
import os
from openai import OpenAI
from dotenv import load_dotenv

import umap
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)
import matplotlib
matplotlib.use('Agg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# ========== 1. Load environment variables ==========
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_key)

# ========== 2. Load Athene Accumulator 10 PDFs ==========
pdf_files = [
    "../data/65223.pdf",  # Accumulator 10 Product Guide
    "../data/65175.pdf",  # Accumulator 10 Product Brochure
]

pdf_texts = []
for file in pdf_files:
    print("reading file ", file)
    reader = PdfReader(file)
    pdf_texts.extend([p.extract_text().strip() for p in reader.pages])

# Remove empty strings
pdf_texts = [text for text in pdf_texts if text]

# ========== 3. Split text into chunks ==========
character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
)
character_split_texts = character_splitter.split_text("\n\n".join(pdf_texts))

token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, tokens_per_chunk=256)
token_split_texts = []
for text in character_split_texts:
    token_split_texts += token_splitter.split_text(text)

# ========== 4. Set up ChromaDB with SentenceTransformer embeddings ==========
embedding_function = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.Client()
chroma_collection = chroma_client.create_collection(
    "athene-accumulator10-collection", embedding_function=embedding_function
)

ids = [str(i) for i in range(len(token_split_texts))]
chroma_collection.add(ids=ids, documents=token_split_texts)
chroma_collection.count()

# ========== 5. Example Query and Retrieval ==========
query = "What are the key features of Athene Accumulator 10 annuity?"

results = chroma_collection.query(query_texts=[query], n_results=5)
retrieved_documents = results["documents"][0]

for document in retrieved_documents:
    print(word_wrap(document))
    print("\n")

# ========== 6. Augment query with a generated answer ==========
def augment_query_generated(query, model="gpt-3.5-turbo"):
    prompt = """You are a helpful insurance product research assistant.
    Provide an example answer to the given question, as it might appear in an annuity product brochure or guide."""
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": query},
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    return content

original_query = "What are the main benefits and surrender charges associated with Athene Accumulator 10?"
hypothetical_answer = augment_query_generated(original_query)
joint_query = f"{original_query} {hypothetical_answer}"
print(word_wrap(joint_query))

results = chroma_collection.query(
    query_texts=[joint_query], n_results=5, include=["documents", "embeddings"]
)
retrieved_documents = results["documents"][0]

# ========== 7. Project Embeddings to 2D for Visualization ==========
embeddings = chroma_collection.get(include=["embeddings"])["embeddings"]
umap_transform = umap.UMAP(random_state=0, transform_seed=0).fit(embeddings)
projected_dataset_embeddings = project_embeddings(embeddings, umap_transform)

retrieved_embeddings = results["embeddings"][0]
original_query_embedding = embedding_function([original_query])
augmented_query_embedding = embedding_function([joint_query])

projected_original_query_embedding = project_embeddings(original_query_embedding, umap_transform)
projected_augmented_query_embedding = project_embeddings(augmented_query_embedding, umap_transform)
projected_retrieved_embeddings = project_embeddings(retrieved_embeddings, umap_transform)

# ========== 8. Plot the projected query and retrieved documents ==========
plt.figure(figsize=(8, 6))
plt.scatter(
    projected_dataset_embeddings[:, 0],
    projected_dataset_embeddings[:, 1],
    s=10,
    color="gray",
    label="All Chunks"
)
plt.scatter(
    projected_retrieved_embeddings[:, 0],
    projected_retrieved_embeddings[:, 1],
    s=100,
    facecolors="none",
    edgecolors="g",
    label="Top Retrieved"
)
plt.scatter(
    projected_original_query_embedding[:, 0],
    projected_original_query_embedding[:, 1],
    s=150,
    marker="X",
    color="r",
    label="Original Query"
)
plt.scatter(
    projected_augmented_query_embedding[:, 0],
    projected_augmented_query_embedding[:, 1],
    s=150,
    marker="X",
    color="orange",
    label="Augmented Query"
)

plt.gca().set_aspect("equal", "datalim")
plt.title(f"Embedding Projection for: {original_query}")
plt.axis("off")
plt.legend()
# (after plotting as before)
plt.savefig("expansion_answer_plot.png", bbox_inches="tight")
print("Plot saved as expansion_answer_plot.png")
# plt.show()  # Omit this line in headless environment