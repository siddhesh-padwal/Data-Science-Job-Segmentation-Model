import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load
df = pd.read_csv('data_science_job.csv')
print("Shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nHead:\n", df.head())
print("\nNulls:\n", df.isnull().sum())
print("\nSalary stats:\n", df['salary_in_usd'].describe())
df['salary_in_usd'].hist(bins=50)
plt.title('Salary Distribution (Raw - Noisy)')
plt.show()

# Cleaning
df_clean = df[df['salary_in_usd'] > 0].dropna(subset=['job_category', 'experience_level', 'company_size'])
print("Clean shape:", df_clean.shape)

# Encode cats
le_exp = LabelEncoder()
le_size = LabelEncoder()
le_cat = LabelEncoder()

df_clean['exp_encoded'] = le_exp.fit_transform(df_clean['experience_level'])
df_clean['size_encoded'] = le_size.fit_transform(df_clean['company_size'])
df_clean['cat_encoded'] = le_cat.fit_transform(df_clean['job_category'])

# Features for clustering
features = ['salary_in_usd', 'exp_encoded', 'size_encoded', 'cat_encoded']
X = df_clean[features]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Features ready. X_scaled shape:", X_scaled.shape)
print("\nExp levels:", dict(zip(le_exp.classes_, le_exp.transform(le_exp.classes_))))
print("Size levels:", dict(zip(le_size.classes_, le_size.transform(le_size.classes_))))
print("Category levels:", dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_))))


# Step 3: Elbow + KMeans (k=4: EN/MI/SE/EX proxies)
inertias = []
K = range(2,11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(K, inertias, 'bx-')
plt.xlabel('Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Fit k=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df_clean['cluster'] = kmeans.fit_predict(X_scaled)

# PCA viz
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df_clean['cluster'], cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Clusters PCA')
plt.colorbar()
plt.show()

# Segment stats
seg_stats = df_clean.groupby('cluster')[features].mean()
print("Cluster stats:\n", seg_stats.round(2))

# Top jobs per cluster
print("\nTop job per cluster:")
for c in range(4):
    top_job = df_clean[df_clean['cluster']==c]['job_title'].value_counts().index[0]
    print(f"Cluster {c}: {top_job}")


# Step 4: Interpret & Export
print("\n=== JOB SEGMENTATION RESULTS ===")
print(df_clean.groupby('cluster').agg({'salary_in_usd':'mean', 'experience_level':'value_counts', 'company_size':'value_counts', 'job_category':'value_counts'}).round(2))

plt.figure(figsize=(10,6))
sns.boxplot(data=df_clean, x='cluster', y='salary_in_usd')
plt.title('Salary Distribution per Segment')
plt.show()

df_clean[['job_title', 'experience_level', 'company_size', 'job_category', 'salary_in_usd', 'cluster']].to_csv('job_segments.csv', index=False)
print("\nSaved job_segments.csv - Your segmentation project complete!")
print("Segments: 0=Entry Eng/ML (in-office), 1=Analyst Remote L-co, 2=Senior Remote, 3=Entry Analysis.")
