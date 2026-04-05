# Data Science Job Segmentation Model Using DBSCAN 

## Overview

This project performs unsupervised clustering on data science job data to segment jobs into meaningful categories based on salary, experience level, job category, employment type, work setting, and company location. The analysis uses DBSCAN clustering with PCA dimensionality reduction, followed by detailed cluster profiling and business insights generation.

## Dataset

The dataset (`data_science_job.csv`) contains 5000 rows and 12 columns with information about data science jobs:

**Columns:**
- `work_year`: Year of the job posting
- `job_title`: Job title (e.g., Data Scientist, Machine Learning Engineer)
- `job_category`: Category (e.g., ML/AI, Data Science, Analysis, Engineering)
- `salary_currency`: Salary currency
- `salary`: Salary amount
- `salary_in_usd`: Salary in USD (primary target for segmentation)
- `employee_residence`: Employee location
- `experience_level`: Experience level (EN=Entry, MI=Mid, SE=Senior, EX=Executive)
- `employment_type`: Employment type (FT=Full-time, PT=Part-time, CT=Contract, FL=Freelance)
- `work_setting`: Work setting (Remote, Hybrid, In-person)
- `company_location`: Company location (country codes)
- `company_size`: Company size (S=Small, M=Medium, L=Large)

**Missing Values:**
- `job_category`: 500 missing
- `salary_currency`: 500 missing
- `experience_level`: 500 missing
- `company_size`: 500 missing
- All other columns: 0 missing

After cleaning (removing rows with missing key categorical values), the dataset contains 4500 rows.

## Methodology

### 1. Data Cleaning
- Remove invalid salary entries (salary ≤ 0)
- Drop rows with missing values in key categorical columns (experience_level, job_category, employment_type, work_setting, company_location)

### 2. Feature Engineering
- Log-transform salary to reduce skewness
- One-hot encode categorical variables (experience_level, job_category, employment_type, work_setting, company_location)
- Final feature matrix: 4500 samples × 18 features

### 3. Dimensionality Reduction
- Apply PCA to reduce 18 features to 3 principal components
- Explained variance: ~23.7% (PC1: 8.3%, PC2: 7.7%, PC3: 7.7%)

### 4. Clustering
- Use DBSCAN clustering on the 3D PCA space
- Parameters: eps=0.5, min_samples=10
- Results: 6 clusters + 41 noise points (0.9% noise)

### 5. Visualization
- PCA scatter plot of clusters (2D projection)
- Noise points shown in gray

### 6. Cluster Profiling
- Analyze salary statistics and categorical distributions for each cluster
- Assign business-friendly segment names based on salary ranges

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

1. Ensure `data_science_job.csv` is in the same directory as the script
2. Run the Python script:
```bash
python job_segmentation_model.py
```

The script will:
- Load and clean the data
- Perform PCA and DBSCAN clustering
- Display visualizations
- Print cluster profiles and business insights
- Export segmented data to `job_segments_dbscan.csv`

## Clustering Results

**Cluster Summary (Salary Statistics):**
| Cluster | Mean Salary | Min Salary | Max Salary |
|---------|-------------|------------|------------|
| -1 (Noise) | $78,507 | $31,265 | $181,477 |
| 0 | $114,086 | $30,016 | $199,985 |
| 1 | $126,220 | $62,716 | $198,711 |
| 2 | $118,543 | $41,857 | $180,390 |
| 3 | $106,588 | $37,943 | $198,560 |
| 4 | $94,517 | $37,025 | $188,607 |

**Cluster Characteristics:**

**Experience Level Distribution:**
- **Noise (-1)**: SE (63.4%), MI (31.7%), EN (4.9%)
- **Cluster 0**: Balanced across EN/MI/SE/EX (~25% each)
- **Cluster 1**: 100% EX (Executive)
- **Cluster 2**: 100% EX (Executive)
- **Cluster 3**: 100% EX (Executive)
- **Cluster 4**: 100% SE (Senior)

**Job Category Distribution:**
- **Noise (-1)**: Data Science (63.4%), Engineering (36.6%)
- **Cluster 0**: Mixed - Data Science/Analysis/ML-AI/Engineering (~25% each)
- **Cluster 1**: 100% ML/AI
- **Cluster 2**: 100% ML/AI
- **Cluster 3**: 100% ML/AI
- **Cluster 4**: 100% Engineering

**Employment Type Distribution:**
- **Noise (-1)**: FL (48.8%), FT (46.3%), CT (4.9%)
- **Cluster 0**: Mixed - PT/CT/FT/FL (~25% each)
- **Cluster 1**: 100% FL (Freelance)
- **Cluster 2**: 100% FL (Freelance)
- **Cluster 3**: 100% FL (Freelance)
- **Cluster 4**: 100% FL (Freelance)

**Work Setting Distribution:**
- **Noise (-1)**: In-person (39%), Hybrid (31.7%), Remote (29.3%)
- **Cluster 0**: Hybrid (34.9%), In-person (33.2%), Remote (31.9%)
- **Cluster 1**: 100% Remote
- **Cluster 2**: 100% In-person
- **Cluster 3**: 100% Hybrid
- **Cluster 4**: 100% Remote

**Company Location Distribution:**
- **Noise (-1)**: UK (97.6%), DE (2.4%)
- **Cluster 0**: Diverse - UK/DE/MX/CN/IN/US (~14% each)
- **Cluster 1**: IN/CN/US (22.2% each), JP (14.8%), MX (11.1%), DE (7.4%)
- **Cluster 2**: DE (23.5%), JP/IN/CN (17.6% each), MX/US (11.8% each)
- **Cluster 3**: DE (38.9%), US (22.2%), MX/CN/IN (11.1% each), JP (5.6%)
- **Cluster 4**: JP (40%), MX (20%), CN/IN (13.3% each), US/DE (6.7% each)

## Output

The script generates:
- PCA cluster visualization with noise points
- Detailed cluster profiles with salary and categorical breakdowns
- Business insights for each segment
- Segmented dataset (`job_segments_dbscan.csv`) with cluster and segment assignments

## Key Insights

The DBSCAN model identifies 6 distinct job market segments:
1. **Mixed Entry-Mid-Senior Roles** (Cluster 0): Balanced experience levels, diverse categories and locations
2. **Executive ML/AI Remote Specialists** (Cluster 1): High-salary executive ML/AI roles, primarily remote freelance
3. **Executive ML/AI In-Person Specialists** (Cluster 2): Executive ML/AI roles requiring in-person work
4. **Executive ML/AI Hybrid Specialists** (Cluster 3): Executive ML/AI roles with hybrid work arrangements
5. **Senior Engineering Remote Specialists** (Cluster 4): Senior engineering roles, remote freelance positions
6. **Noise/Outliers** (Cluster -1): Atypical job profiles that don't fit the main clusters

This segmentation helps:
- Job seekers understand market positioning by role type and requirements
- Companies identify competitive salary ranges for specific role profiles
- Recruiters target candidates with matching experience and work preferences
- Career planners navigate different specialization paths in data science

## Files

- `job_segmentation_model.py`: Main analysis script
- `data_science_job.csv`: Input dataset (5000 rows, 12 columns)
- `job_segments_dbscan.csv`: Output with cluster assignments (generated)