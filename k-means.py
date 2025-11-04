import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer  # A useful tool for the Elbow Method

# --- 1. Data Loading and Preparation ---

# Generate synthetic 2D data with 3 distinct clusters
N_SAMPLES = 500
TRUE_K = 3
X, y_true = make_blobs(
    n_samples=N_SAMPLES,
    centers=TRUE_K,
    cluster_std=0.8,  # Tightness of clusters
    random_state=42,
)
df = pd.DataFrame(X, columns=["Feature_1", "Feature_2"])

# In unsupervised learning, we typically only have the feature matrix (X)
matrix = df
# Define features for transformation
num_standard_cols = ["Feature_1", "Feature_2"]

print("--- 1. Data Preparation ---")
print(f"Dataset shape: {matrix.shape}")

# --- 2. Feature Preprocessing Pipeline ---

# For clustering, scaling is critical as K-Means is distance-based.
standard_scaler = StandardScaler().set_output(transform="pandas")

# Define the full preprocessing pipeline
preprocessor = Pipeline(
    steps=[
        ("scale_standard", standard_scaler)  # Only scaling needed for this clean data
    ]
).set_output(transform="pandas")

# Fit and transform the data
X_processed = preprocessor.fit_transform(matrix)
print(f"Processed feature shape: {X_processed.shape}")

# --- 3. Determining Optimal K using the Elbow Method ---

print("\n--- 3. Determining Optimal K (Elbow Method) ---")

# We will test K from 2 up to 10
K_RANGE = range(2, 11)
inertia_list = []

# Create a pipeline that combines preprocessing and the K-Means model
kmeans_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "kmeans",
            KMeans(n_init="auto", random_state=42),
        ),  # 'n_init' added for sklearn best practices
    ]
)

# Fit the pipeline for each potential K
for k in K_RANGE:
    kmeans_pipeline.set_params(
        kmeans__n_clusters=k
    )  # Set the number of clusters for the K-Means step
    kmeans_pipeline.fit(matrix)
    inertia_list.append(kmeans_pipeline.named_steps["kmeans"].inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(8, 4))
plt.plot(K_RANGE, inertia_list, marker="o", linestyle="--")
plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-cluster Sum of Squares)")
plt.grid(True)
# The "elbow" point is where the reduction in inertia starts to slow significantly (here, K=3)
plt.show()

# --- 4. Final Model Training and Prediction ---

# Based on the Elbow plot, we select the optimal K=3
OPTIMAL_K = 3

print(f"\n--- 4. Final Model Training with K={OPTIMAL_K} ---")

# Update the pipeline with the optimal K
kmeans_pipeline.set_params(kmeans__n_clusters=OPTIMAL_K)

# Fit the final model to the processed data
kmeans_pipeline.fit(matrix)

# Get cluster assignments (labels) and cluster centers
cluster_labels = kmeans_pipeline.predict(matrix)
cluster_centers = kmeans_pipeline.named_steps["kmeans"].cluster_centers_

# --- 5. Evaluation and Visualization ---

# Calculate the Silhouette Score (measures how similar an object is to its own cluster
# compared to other clusters. Higher is better.)
score = silhouette_score(X_processed, cluster_labels)
print(f"Final Model Silhouette Score: {score:.4f}")

# Visualize the final clustering
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    df["Feature_1"],
    df["Feature_2"],
    c=cluster_labels,
    cmap="viridis",
    marker="o",
    edgecolor="k",
)
# Plot cluster centers
plt.scatter(
    cluster_centers[:, 0],
    cluster_centers[:, 1],
    c="red",
    s=200,
    marker="X",
    label="Cluster Centers",
)
plt.title(f"K-Means Clustering (K={OPTIMAL_K})")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.show()

# Add cluster labels back to the original DataFrame for analysis
df["Cluster"] = cluster_labels
print("\nFirst 5 rows with Cluster Labels:")
print(df.head())
