import requests
import re
from ftfy import fix_text
from sentence_transformers import SentenceTransformer, util
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2') 

# Download the text from Gutenberg
book = requests.get("https://www.gutenberg.org/cache/epub/1342/pg1342.txt").text

# Preprocessing step
# Remove text before chapter 1
book = book.split("Chapter I")[-1]

# Split book into passages
passages = re.split('[\n\r]{3,}', book)
passages = [fix_text(chunk.strip()) for chunk in passages if len(chunk.strip()) > 100]

# Vectorise the passages
passage_embeddings = model.encode(passages, convert_to_tensor=True)
print("-" * 50)

# Query against the embeddings
query_text = [
    "a woman looks for a husband"
]

# Convert query into a vector using the same model as above
query_embeddings = model.encode(query_text, convert_to_tensor=True)

# Calculate cosine similarity
cos_scores = util.cos_sim(query_embeddings, passage_embeddings)[0]
top_results = cos_scores.topk(k=5)

# Print results
for score, idx in zip(top_results.values, top_results.indices):
    print(f"Match Confidence Score: {score:.3f}")
    print(f"Passage:\n{passages[idx]}")
    print("-" * 50)


# Create a visualisation of key themes in the book
print("\nMapping multi-theme semantic clusters")

themes = {
    "Marriage": "romantic proposals wedding matrimony wife husband",
    "Pride": "arrogance pride dignity stubborn superior social status",
    "Money": "inheritance wealth fortune income pounds estate rich",
    "Scandal": "gossip drama talking secrets rumours shame disgrace elopement"
}

# 2. Define colors for each theme
colour_map = {
    "Marriage": "pink", 
    "Pride": "green", 
    "Money": "blue", 
    "Scandal": "yellow"
}
# Convert to numpy format for scikit-learn
all_vectors_np = passage_embeddings.cpu().numpy()

# Flatten the dimensions of the book to 2D coordinates
pca = PCA(n_components=2)
coordinates_2d = pca.fit_transform(all_vectors_np)

plt.figure(figsize=(12, 8))

# Create scatterplot with every passage in the book documented as a small grey dot
plt.scatter(coordinates_2d[:, 0], coordinates_2d[:, 1], 
            c='lightgrey', alpha=0.5, s=15, label='Other Book Passages')

for theme_name, query_phrase in themes.items():
    theme_embeddings = model.encode(query_phrase, convert_to_tensor=True)

    theme_scores = util.cos_sim(theme_embeddings, passage_embeddings)[0]
    top_results = theme_scores.topk(k=10)
    top_indices = top_results.indices.cpu().numpy()

    theme_x = coordinates_2d[top_indices, 0]
    theme_y = coordinates_2d[top_indices, 1]

    plt.scatter(theme_x, theme_y, 
            c=colour_map[theme_name],
            edgecolors='black',
            s=120,
            zorder=5,
            label=theme_name)

# Chart styles
plt.title(f"Semantic Clusters in Pride and Prejudice", fontsize=16, fontweight='bold')
plt.xlabel("Semantic Dimension X")
plt.ylabel("Semantic Dimension Y")
plt.legend(title="Themes", fontsize=11)
plt.grid(True, linestyle='--', alpha=0.3)

plt.show()



