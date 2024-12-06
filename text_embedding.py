import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import pickle

# Constants
DATA_DIR = "./result_analysis_chatgpt-4o-latest/1_shot_v7.0_0.7_2row"
EXCEL_FILE = os.path.join(DATA_DIR, "case_analysis_datagram.xlsx")
MODEL_NAME = "jinaai/jina-embeddings-v3"
OUTPUT_EMBEDDINGS = os.path.join(DATA_DIR, "thoughts_embeddings.pkl")

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Load the Excel file
thoughts_df = pd.read_excel(EXCEL_FILE, sheet_name="Thoughts Table")

# Function to compute embeddings
def compute_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean pooling
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

# Compute embeddings for each thought
embeddings = []
for i, thought in enumerate(thoughts_df["Thoughts"]):
    embedding = compute_embedding(thought)
    embeddings.append(embedding)
    print(f"Processed {i + 1}/{len(thoughts_df)}")

# Prepare data for saving
embedding_data = {
    "Case Name": thoughts_df["Case Name"].tolist(),
    "Rep": thoughts_df["Rep"].tolist(),
    "Embedding": embeddings
}

# Save to Pickle format
with open(OUTPUT_EMBEDDINGS, "wb") as f:
    pickle.dump(embedding_data, f)
print(f"Saved embeddings to {OUTPUT_EMBEDDINGS}")

# Preview the saved data
print(f"Embedding data structure: {list(embedding_data.keys())}")
print(f"First few case names: {embedding_data['Case Name'][:5]}")
print(f"First few reps: {embedding_data['Rep'][:5]}")
print(f"First embedding sample: {embedding_data['Embedding'][0][:5]}...")
