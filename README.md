# 🧬 Breast Cancer Clustering using K-Means

This project explores how unsupervised learning can be used to identify patterns in medical data. K-Means clustering is applied to the Breast Cancer dataset, and the results are visualized using PCA to compare with the actual diagnoses.

# 📌 Overview
- Used the Wisconsin Breast Cancer dataset
- Applied K-Means clustering to group the data into 2 clusters
- Reduced dimensions using PCA for visualization
- Compared cluster results with actual labels (benign vs malignant)

# ⚙️ Tech Stack
- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

# 📊 Dataset Details
- Total samples: 569
- Features: 30 numerical features
- Target classes:
- 0 → Malignant
- 1 → Benign

# 🚀 How it works
- Load dataset using load_breast_cancer()
- Standardize the features
- Apply K-Means clustering (k = 2)
- Use PCA to reduce data to 2 dimensions
- Visualize clusters and compare with actual labels

# 📈 Results
- The clustering roughly separates malignant and benign cases
- Some overlap is observed due to the unsupervised nature of K-Means
- PCA helps in visualizing how well the clusters align with actual classes

# Observations
- K-Means does not use actual labels, so results are not perfectly aligned
- Standardization is important for better clustering
- PCA makes high-dimensional data easier to interpret visually

# ▶️ How to Run
- pip install numpy pandas matplotlib seaborn scikit-learn
- python your_file_name.py
