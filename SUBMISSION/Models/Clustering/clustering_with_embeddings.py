import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# =======================
# Paths
# =======================
# Get absolute path of current script directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data folder (CSV files)
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "..", "Final Data"))
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV  = os.path.join(DATA_DIR, "test.csv")

# Results folder
CLUSTER_DIR = os.path.abspath(os.path.join(BASE_DIR, "results", "with_embeddings"))
os.makedirs(CLUSTER_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(CLUSTER_DIR, "clustered_tweets.csv")

# Embeddings folder
EMB_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "CNN", "embeddings", "cnn"))

# =======================
# Load datasets
# =======================
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

emb_train = np.load(os.path.join(EMB_DIR, "emb_train.npy"))
emb_test  = np.load(os.path.join(EMB_DIR, "emb_test.npy"))

assert len(train_df) == emb_train.shape[0], "emb_train.npy and train.csv are misaligned"
assert len(test_df)  == emb_test.shape[0],  "emb_test.npy and test.csv are misaligned"

# =======================
# Preprocess text
# =======================
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

test_df["text_clean"] = test_df["text"].apply(preprocess_text)

# =======================
# Split by label
# =======================
idx_fake = test_df["fake"] == 1
idx_real = ~idx_fake

X_fake = emb_test[idx_fake.values]
X_real = emb_test[idx_real.values]

fake_texts = test_df.loc[idx_fake, "text_clean"].tolist()
real_texts = test_df.loc[idx_real, "text_clean"].tolist()

# =======================
# Clustering helper
# =======================
def cluster_and_evaluate(X, texts, n_clusters=3, theme_map=None, title="Clusters"):
    if X.shape[0] == 0:
        print(f"No items in {title}")
        return None, None

    # Scale embeddings
    Xs = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(Xs)

    # Cluster sizes
    print(f"\n--- Items per Cluster ({title}) ---")
    for i in range(n_clusters):
        count = int((clusters == i).sum())
        label = theme_map[i] if theme_map else f"Cluster {i}"
        print(f"{label}: {count} items")

    # Evaluation
    sse = float(kmeans.inertia_)
    sil_score = silhouette_score(Xs, clusters) if Xs.shape[0] > 1 else float("nan")
    db_score  = davies_bouldin_score(Xs, clusters) if Xs.shape[0] > 1 else float("nan")

    print(f"\n=== {title} Evaluation ===")
    print(f"SSE: {sse:.2f}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Score: {db_score:.4f}")

    # PCA 2D visualization
    if Xs.shape[0] > 1:
        X_reduced = PCA(n_components=2, random_state=42).fit_transform(Xs)
        plt.figure(figsize=(8,6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="tab10", alpha=0.7)
        for i in range(n_clusters):
            mask = clusters == i
            if not np.any(mask):
                continue
            center = X_reduced[mask].mean(axis=0)
            label = theme_map[i] if theme_map else f"Cluster {i}"
            plt.text(center[0], center[1], label, fontsize=12, fontweight="bold",
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        plt.title(f"{title} (PCA 2D on CNN embeddings)")
        plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
        plt.colorbar(label="Cluster")
        plt_path = os.path.join(CLUSTER_DIR, f"{title.replace(' ','_')}_PCA.png")
        plt.savefig(plt_path); plt.close()
        print(f"Saved PCA plot: {plt_path}")

    # Map clusters to theme labels
    theme_labels = [theme_map[c] if theme_map else c for c in clusters]
    return clusters, theme_labels

# =======================
# Run clustering
# =======================
fake_theme_map = {0: "Anti-Vax", 1: "Fake Cures", 2: "Conspiracy"}
real_theme_map = {0: "Official Health Advice", 1: "Scientific Findings", 2: "News / Reports"}

clusters_fake, themes_fake = cluster_and_evaluate(X_fake, fake_texts, n_clusters=3, theme_map=fake_theme_map, title="Fake Tweets")
clusters_real, themes_real = cluster_and_evaluate(X_real, real_texts, n_clusters=3, theme_map=real_theme_map, title="Real Tweets")

# =======================
# Save clustered CSV
# =======================
df_fake = test_df[idx_fake].copy()
df_fake["theme"] = themes_fake

df_real = test_df[idx_real].copy()
df_real["theme"] = themes_real

df_clustered = pd.concat([df_fake, df_real], ignore_index=True)
df_clustered.to_csv(OUTPUT_CSV, index=False)
print(f"\nClustered dataset saved to: {OUTPUT_CSV}")
