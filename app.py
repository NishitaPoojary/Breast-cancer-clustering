import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler

# Load dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
actual_labels = data.target  # 0 = malignant, 1 = benign

print(df.head())
print(df.info())
print(df.shape)

# Standardize data
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Apply K-Means
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(df_scaled)
cluster_labels = kmeans.labels_

# PCA for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Create visualization dataframe
df_visual = pd.DataFrame(df_pca, columns=['PC1', 'PC2'])
df_visual['Cluster'] = cluster_labels
df_visual['Actual'] = actual_labels

# Plot
plt.figure(figsize=(12, 5))

# K-Means Clusters
plt.subplot(1, 2, 1)
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_visual, palette='Set1')
plt.title('K-Means Clustering')

# Actual Labels
plt.subplot(1, 2, 2)
sns.scatterplot(x='PC1', y='PC2', hue='Actual', data=df_visual, palette='Set2')
plt.title('Actual Diagnosis')

plt.tight_layout()
plt.show()