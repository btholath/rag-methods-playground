from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    DPRQuestionEncoderTokenizer,
    DPRContextEncoderTokenizer,
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained DPR models and tokenizers
question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

# Encode a life insurance-related query
query = "What are the benefits of whole life insurance?"
question_inputs = question_tokenizer(query, return_tensors="pt")
question_embedding = question_encoder(**question_inputs).pooler_output

# Encode life insurance-related passages
passages = [
    "Whole life insurance provides lifelong coverage with a guaranteed death benefit.",
    "Term life insurance is cheaper but only covers a fixed period, like 10 or 20 years.",
    "Universal life insurance offers flexible premiums and the ability to build cash value.",
    "Beneficiaries receive a tax-free payout upon the policyholder's death.",
    "Many whole life policies accumulate cash value over time that can be borrowed against.",
    "Life insurance is essential for providing financial security to dependents.",
    "Certain policies allow policyholders to access funds in case of terminal illness.",
    "Premiums vary based on age, health, and coverage amount.",
]

context_embeddings = []
for passage in passages:
    context_inputs = context_tokenizer(passage, return_tensors="pt")
    context_embedding = context_encoder(**context_inputs).pooler_output
    context_embeddings.append(context_embedding)

context_embeddings = torch.cat(context_embeddings, dim=0)

# Compute similarities between query and passages
similarities = cosine_similarity(
    question_embedding.detach().numpy(), context_embeddings.detach().numpy()
)
print("Similarities:", similarities)

# Get the most relevant life insurance-related passage
most_relevant_idx = np.argmax(similarities)
print("Most relevant passage:", passages[most_relevant_idx])