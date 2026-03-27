# ===============================
# 1. IMPORTS
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
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
    'company_size',
    'job_category'
])

print("\nClean shape:", df_clean.shape)


# ===============================
# 4. FEATURE ENGINEERING
# ===============================

# Log transform (reduces skew)
df_clean['log_salary'] = np.log1p(df_clean['salary_in_usd'])

# One-hot encoding (IMPORTANT FIX)
cat_cols = ['experience_level', 'company_size', 'job_category']
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
# 6. FIND OPTIMAL K (ELBOW + SILHOUETTE)
# ===============================
inertias = []
sil_scores = []

K = range(2, 10)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    inertias.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(K, inertias, 'bx-')
plt.title("Elbow Method")

plt.subplot(1,2,2)
plt.plot(K, sil_scores, 'gx-')
plt.title("Silhouette Score")

plt.show()

# Choose best k
best_k = K[np.argmax(sil_scores)]
print("Best k:", best_k)


# ===============================
# 7. FINAL CLUSTERING
# ===============================
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)


# ===============================
# 8. PCA VISUALIZATION
# ===============================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=df_clean['cluster'],
    palette='viridis'
)
plt.title("Cluster Visualization (PCA)")
plt.show()


# ===============================
# 9. CLUSTER PROFILING (KEY PART)
# ===============================
profile = df_clean.groupby('cluster').agg({
    'salary_in_usd': ['mean','min','max'],
    'experience_level': lambda x: x.mode()[0],
    'company_size': lambda x: x.mode()[0],
    'job_category': lambda x: x.mode()[0]
})

print("\nCluster Profile:\n", profile)


# ===============================
# 10. SMART SEGMENT NAMING
# ===============================
cluster_names = {}

for c in df_clean['cluster'].unique():
    avg_salary = df_clean[df_clean['cluster']==c]['salary_in_usd'].mean()
    
    if avg_salary < 70000:
        cluster_names[c] = "Entry-Level / Low Salary"
    elif avg_salary < 120000:
        cluster_names[c] = "Mid-Level Professionals"
    else:
        cluster_names[c] = "Senior / High Salary"

df_clean['segment'] = df_clean['cluster'].map(cluster_names)


# ===============================
# 11. BUSINESS INSIGHTS
# ===============================
print("\n=== BUSINESS INSIGHTS ===")

for c in df_clean['cluster'].unique():
    subset = df_clean[df_clean['cluster']==c]
    
    print(f"\nSegment: {cluster_names[c]}")
    print("Avg Salary:", round(subset['salary_in_usd'].mean(),2))
    print("Top Role:", subset['job_title'].value_counts().index[0])
    print("Top Category:", subset['job_category'].value_counts().index[0])
    print("Top Company Size:", subset['company_size'].value_counts().index[0])


# ===============================
# 12. VISUAL INSIGHTS
# ===============================
plt.figure(figsize=(10,6))
sns.boxplot(data=df_clean, x='segment', y='salary_in_usd')
plt.xticks(rotation=20)
plt.title("Salary Distribution by Segment")
plt.show()


# ===============================
# 13. EXPORT
# ===============================
df_clean[['job_title','experience_level','company_size','job_category',
          'salary_in_usd','cluster','segment']].to_csv('job_segments_v2.csv', index=False)

print("\n✅ Project Complete! File saved: job_segments_v2.csv")