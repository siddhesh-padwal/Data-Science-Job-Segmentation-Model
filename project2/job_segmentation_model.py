# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

import warnings
warnings.filterwarnings('ignore')


# ===============================
# 2. LOAD + BASIC EDA
# ===============================
df = pd.read_csv('data_science_job.csv')

print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nMissing values:\n", df.isnull().sum())

# Salary distribution
plt.figure(figsize=(8,5))
df['salary_in_usd'].hist(bins=50)
plt.title('Raw Salary Distribution')
plt.show()


# ===============================
# 3. CLEANING
# ===============================
df = df[df['salary_in_usd'] > 0]

df_clean = df.dropna(subset=[
    'experience_level',
    'job_category',
    'employment_type',
    'work_setting',
    'company_location'
]).copy().reset_index(drop=True)

print("\nClean shape:", df_clean.shape)


# ===============================
# 4. FEATURE ENGINEERING
# ===============================

# Log transform (reduces skew)
df_clean['log_salary'] = np.log1p(df_clean['salary_in_usd'])

# One-hot encoding (IMPORTANT FIX)
cat_cols = ['experience_level', 'job_category', 'employment_type', 'work_setting', 'company_location']
df_encoded = pd.get_dummies(df_clean[cat_cols], drop_first=True)

# Final feature set
X = pd.concat([df_clean[['log_salary']], df_encoded], axis=1)

print("\nFeature shape:", X.shape)


# ===============================
# 5. SCALING
# ===============================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ===============================
# 6. PCA FOR DIMENSIONALITY REDUCTION
# ===============================
pca = PCA(n_components=3)  # Reduce to 3 principal components for DBSCAN
X_pca = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_)}")


# ===============================
# 7. DBSCAN CLUSTERING PARAMETERS
# ===============================
# DBSCAN parameters (tune as needed based on data density)
eps = 0.5  # Neighborhood distance (smaller in reduced dimensions)
min_samples = 10  # Minimum samples in neighborhood

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df_clean['cluster'] = dbscan.fit_predict(X_pca)

# Check number of clusters (excluding noise: -1)
n_clusters = len(set(df_clean['cluster'])) - (1 if -1 in df_clean['cluster'] else 0)
n_noise = list(df_clean['cluster']).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")


# 8. PCA VISUALIZATION
plt.figure(figsize=(8,6))
# Filter out noise points for cleaner visualization (optional)
mask = df_clean['cluster'] != -1
sns.scatterplot(
    x=X_pca[mask, 0],
    y=X_pca[mask, 1],
    hue=df_clean[mask]['cluster'],
    palette='viridis'
)
# Plot noise points in gray
if not mask.all():
    plt.scatter(X_pca[~mask, 0], X_pca[~mask, 1], c='gray', label='Noise', alpha=0.5)
plt.title("Cluster Visualization (PCA) - DBSCAN")
plt.legend()
plt.show()


cluster_summary = df_clean.groupby('cluster').agg({
    'salary_in_usd': ['mean', 'min', 'max'],
}).round(2)

print(cluster_summary)

for col in cat_cols:
    print(f"\n--- {col} ---")
    print(df_clean.groupby('cluster')[col].value_counts(normalize=True))