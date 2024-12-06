import os
import pickle
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Constants
DATA_DIR = "./result_analysis_chatgpt-4o-latest/1_shot_v7.0_0.7_2row"
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "thoughts_embeddings.pkl")
EXCEL_FILE = os.path.join(DATA_DIR, "case_analysis_datagram.xlsx")
OUTPUT_DIR = os.path.join(DATA_DIR, "tsne_plots")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load embeddings and metadata
with open(EMBEDDINGS_FILE, "rb") as f:
    embedding_data = pickle.load(f)

case_names = embedding_data["Case Name"]
reps = embedding_data["Rep"]
embeddings = np.array(embedding_data["Embedding"])

# Load difficulty levels from the Case Analysis sheet
case_analysis = pd.read_excel(EXCEL_FILE, sheet_name="Case Analysis")
level_mapping = dict(zip(case_analysis["Case Name"], case_analysis["Level"]))
difficulty_colors = {"Hard": "red", "Intermediate": "blue", "Easy": "green"}

# Perform t-SNE
print("Performing t-SNE...")
tsne = TSNE(n_components=2, random_state=42, init="random", perplexity=30)
embeddings_2d = tsne.fit_transform(embeddings)
print("t-SNE completed.")

# Generate 16 plots (one for each case)
for focused_case in set(case_names):
    print(f"Generating t-SNE plot for {focused_case}...")

    # Identify indices for the focused case and background
    focused_indices = [i for i, name in enumerate(case_names) if name == focused_case]
    background_indices = [i for i, name in enumerate(case_names) if name != focused_case]

    # Prepare plot
    plt.figure(figsize=(12, 8))
    
    # Plot background points
    plt.scatter(
        embeddings_2d[background_indices, 0],
        embeddings_2d[background_indices, 1],
        c="lightgray",
        label="Background",
        alpha=0.5,
        s=50
    )

    # Plot focused points with colors based on difficulty
    for idx in focused_indices:
        level = level_mapping[case_names[idx]]
        plt.scatter(
            embeddings_2d[idx, 0],
            embeddings_2d[idx, 1],
            c=difficulty_colors[level],
            label=level if level not in plt.gca().get_legend_handles_labels()[1] else None,
            edgecolors="black",
            s=150
        )

    # Add title, legend, and save plot
    plt.title(f"t-SNE Visualization for {focused_case}", fontsize=16)
    plt.legend(title="Difficulty Level", loc="upper right")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.tight_layout()

    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, f"tsne_{focused_case}.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved t-SNE plot to {plot_path}")

print("All t-SNE plots generated.")
