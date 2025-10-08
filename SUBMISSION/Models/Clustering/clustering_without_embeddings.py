import os
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
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
CLUSTER_DIR = os.path.abspath(os.path.join(BASE_DIR, "results", "without_embeddings"))
os.makedirs(CLUSTER_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(CLUSTER_DIR, "clustered_tweets.csv")


# Helper: text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)            # remove mentions
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()    # remove extra whitespace
    return text

# Load datasets
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_texts = train_df["text"].apply(preprocess_text).tolist()
test_df["text_clean"] = test_df["text"].apply(preprocess_text)

fake_tweets = test_df[test_df["fake"] == 1]["text_clean"].tolist()
real_tweets = test_df[test_df["fake"] == 0]["text_clean"].tolist()

# Custom stopwords for COVID
covid_stopwords = [
    "covid", "coronavirus", "virus", "pandemic", "sarscov2", "covid19"
]

# Combine English stopwords with custom COVID words
combined_stopwords = list(set(ENGLISH_STOP_WORDS).union(covid_stopwords))

# Fit TF-IDF on train data
vectorizer = TfidfVectorizer(
    stop_words=combined_stopwords,
    max_features=5000,
    ngram_range=(1,2)
)
vectorizer.fit(train_texts)


# Clustering and evaluation
def cluster_and_evaluate(tweets, n_clusters=3, theme_map=None, title="Clusters"):
    if not tweets:
        print(f"No tweets in {title}")
        return None, None

    X = vectorizer.transform(tweets)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    # Top keywords per cluster
    terms = vectorizer.get_feature_names_out()
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    print(f"\n--- Top Keywords per Cluster ({title}) ---")
    for i in range(n_clusters):
        top_terms = [terms[ind] for ind in order_centroids[i, :10]]
        print(f"Cluster {i}: {', '.join(top_terms)}")

    # Tweets per cluster
    print(f"\n--- Tweets per Cluster ({title}) ---")
    cluster_counts = {}
    for i in range(n_clusters):
        count = sum(clusters == i)
        label = theme_map[i] if theme_map else f"Cluster {i}"
        cluster_counts[label] = count
        print(f"{label}: {count} tweets")

    # Cluster evaluation
    sse = kmeans.inertia_
    sil_score = silhouette_score(X, clusters) if len(tweets) > 1 else float("nan")
    db_score = davies_bouldin_score(X.toarray(), clusters) if len(tweets) > 1 else float("nan")

    print(f"\n=== {title} Evaluation ===")
    print(f"SSE: {sse:.2f}")
    print(f"Silhouette Score: {sil_score:.4f}")
    print(f"Davies-Bouldin Score: {db_score:.4f}")

    # PCA visualization
    if len(tweets) > 1:
        X_reduced = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
        plt.figure(figsize=(8,6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap="tab10", alpha=0.7)
        for i in range(n_clusters):
            cluster_points = X_reduced[clusters == i]
            if len(cluster_points) == 0:
                continue
            center = cluster_points.mean(axis=0)
            label = theme_map[i] if theme_map else f"Cluster {i}"
            plt.text(center[0], center[1], label, fontsize=12, fontweight="bold",
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
        plt.title(f"{title} (PCA 2D)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(label="Cluster")
        plt_path = os.path.join(CLUSTER_DIR, f"{title.replace(' ','_')}_pca.png")
        plt.savefig(plt_path)
        plt.close()

    # Map clusters to theme labels
    theme_labels = [theme_map[c] if theme_map else c for c in clusters]

    return clusters, theme_labels

def plot_elbow_and_silhouette(X, max_k=8, title="Clusters", save_path=None):
    sse = []
    silhouette_scores = []

    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        sse.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    fig, ax1 = plt.subplots(figsize=(10,6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of clusters k')
    ax1.set_ylabel('SSE', color=color)
    ax1.plot(range(2, max_k+1), sse, marker='o', color=color, label='SSE')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_xticks(range(2, max_k+1))

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Score', color=color)
    ax2.plot(range(2, max_k+1), silhouette_scores, marker='o', color=color, label='Silhouette')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.title(f"{title} Elbow and Silhouette Scores")

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved elbow/silhouette plot: {save_path}")
    else:
        plt.show()


def plot_clusters_pca(X, clusters, theme_map=None, title="Clusters", save_path=None):
    if X.shape[0] <= 1:
        return

    X_reduced = PCA(n_components=2, random_state=42).fit_transform(X.toarray())
    plt.figure(figsize=(8,6))
    plt.scatter(X_reduced[:,0], X_reduced[:,1], c=clusters, cmap="tab10", alpha=0.7)

    n_clusters = len(set(clusters))
    for i in range(n_clusters):
        cluster_points = X_reduced[clusters == i]
        if len(cluster_points) == 0:
            continue
        center = cluster_points.mean(axis=0)
        label = theme_map[i] if theme_map else f"Cluster {i}"
        plt.text(center[0], center[1], label, fontsize=12, fontweight="bold",
                 bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))

    plt.title(f"{title} (PCA 2D)")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.colorbar(label="Cluster")

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved cluster plot: {save_path}")
    else:
        plt.show()


# Theme mappings
fake_theme_map = {0: "Anti-Vax", 1: "Fake Cures", 2: "Conspiracy"}
real_theme_map = {0: "Official Health Advice", 1: "Scientific Findings", 2: "News / Reports"}

# Run clustering
clusters_fake, themes_fake = cluster_and_evaluate(fake_tweets, n_clusters=3, theme_map=fake_theme_map, title="Fake Tweet Clusters")
clusters_real, themes_real = cluster_and_evaluate(real_tweets, n_clusters=3, theme_map=real_theme_map, title="Real Tweet Clusters")

# Save clustered CSV
df_fake = test_df[test_df["fake"] == 1].copy()
df_fake["theme"] = themes_fake
df_real = test_df[test_df["fake"] == 0].copy()
df_real["theme"] = themes_real

df_clustered = pd.concat([df_fake, df_real], ignore_index=True)
df_clustered.to_csv(OUTPUT_CSV, index=False)
print(f"\nClustered dataset saved to: {OUTPUT_CSV}")