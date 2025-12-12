import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 60)
print("DENGUE RISK PREDICTION USING K-MEANS CLUSTERING")
print("=" * 60)

# ****************************************************
# 1. LOAD DATA

print("\n[1] Loading Data...")

# Load the dataset (adjust the path to your CSV file)
df = pd.read_csv('dengue_2016_2024_combined.csv')  # Change this to your file path


# Handle missing values (if any)
# Forward fill for time series data
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')

# Remove any remaining rows with missing values
df = df.dropna()

print(f"\nDataset Shape: {df.shape}")

# ******************************************************
# 2. FEATURE SELECTION AND SCALING

print("\n [2] Feature Selection and Scaling...")

# Define the features for clustering
# Based on your dataset columns
feature_columns = ['Dengue_Cases', 'Dengue_Deaths', 'Temperature_C',
                   'Humidity_pct', 'Rainfall_mm']

df_features = df[feature_columns].copy()

print(f"\nSelected Features: {feature_columns}")
print(f"\nFeature Statistics:")
print(df_features.describe())

scaler = StandardScaler()

df_scaled = scaler.fit_transform(df_features)

df_scaled = pd.DataFrame(df_scaled, columns=feature_columns)

print(f"\nScaled Data (first 5 rows):")
print(df_scaled.head())

print(f"\nScaled Data Statistics (mean should be ~0, std should be ~1):")
print(df_scaled.describe())

# ****************************************************
# 3. K-MEANS CLUSTERING

print("\n[3] Applying K-Means Clustering (k=3)...")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)

clusters = kmeans.fit_predict(df_scaled)

df['Cluster'] = clusters

print(f"\nCluster Distribution:")
print(df['Cluster'].value_counts().sort_index())

# ****************************************************
# 4. CALCULATE RISK SCORES (1-100)

print("\n[4] Calculating Risk Scores (1-100)...")


distances = np.min(kmeans.transform(df_scaled), axis=1)

max_distance = distances.max()
min_distance = distances.min()

distance_risk_normalized = (distances - min_distance) / (max_distance - min_distance)

# Normalize actual dengue cases to 0-1 range
dengue_cases_normalized = (df['Dengue_Cases'] - df['Dengue_Cases'].min()) / \
                          (df['Dengue_Cases'].max() - df['Dengue_Cases'].min())

# Normalize dengue deaths to 0-1 range
dengue_deaths_normalized = (df['Dengue_Deaths'] - df['Dengue_Deaths'].min()) / \
                           (df['Dengue_Deaths'].max() - df['Dengue_Deaths'].min() + 1e-10)

# Combine multiple factors for comprehensive risk score
# Weights: 50% dengue cases, 25% deaths, 25% cluster distance
risk_scores_normalized = (0.50 * dengue_cases_normalized +
                         0.25 * dengue_deaths_normalized +
                         0.25 * distance_risk_normalized)

# Scale to 1-100 range
risk_scores = 1 + (risk_scores_normalized * 99)

# Ensure risk scores are between 1 and 100
risk_scores = np.clip(risk_scores, 1, 100)

df['Risk_Score'] = risk_scores.round(2)

# Categorize risk based on score
def categorize_risk(score):
    if score >= 50:
        return 'High Risk (50-100)'
    elif score >= 20:
        return 'Medium Risk (20-50)'
    else:
        return 'Low Risk (0-20)'

df['Risk_Category'] = df['Risk_Score'].apply(categorize_risk)

print(f"\nRisk Category Distribution:")
print(df['Risk_Category'].value_counts())

print(f"\nRisk Score Statistics:")
print(df['Risk_Score'].describe())

# Display risk score range for each category
print(f"\nRisk Score Ranges by Category:")
for category in ['Low Risk (0-40)', 'Medium Risk (40-70)', 'High Risk (70-100)']:
    category_scores = df[df['Risk_Category'] == category]['Risk_Score']
    if len(category_scores) > 0:
        print(f"  {category}:")
        print(f"    Min: {category_scores.min():.2f}, Max: {category_scores.max():.2f}, "
              f"Mean: {category_scores.mean():.2f}")

# ***************************************************
# 5. MODEL EVALUATION

print("\n[5] Evaluating Model Performance...")

# Calculate Silhouette Score
silhouette_avg = silhouette_score(df_scaled, clusters)
print(f"\nSilhouette Score: {silhouette_avg:.4f}")
print("(Range: -1 to 1, higher is better, >0.5 is good)")

# Calculate Davies-Bouldin Index
davies_bouldin = davies_bouldin_score(df_scaled, clusters)
print(f"\nDavies-Bouldin Index: {davies_bouldin:.4f}")
print("(Lower is better, <1 is good)")

# Calculate Calinski-Harabasz Score
calinski_harabasz = calinski_harabasz_score(df_scaled, clusters)
print(f"\nCalinski-Harabasz Score: {calinski_harabasz:.4f}")
print("(Higher is better)")


#  *********************************************
# 6. PCA VISUALIZATION

print("\n[6] Creating PCA Visualization...")

# Apply PCA to reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Create the plot
plt.figure(figsize=(14, 10))

# Plot each cluster with different colors
colors = ['#4ECDC4', '#FFD93D', '#FF6B6B']  # Low, Medium, High risk colors
cluster_labels = ['Cluster 0', 'Cluster 1', 'Cluster 2']

for i in range(3):
    cluster_points = df_pca[clusters == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1],
                c=colors[i], label=cluster_labels[i],
                alpha=0.6, s=120, edgecolors='black', linewidth=0.5)

# Transform and plot centroids
centroids_pca = pca.transform(kmeans.cluster_centers_)
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            c='darkred', marker='X', s=600,
            edgecolors='black', linewidth=2.5,
            label='Centroids', zorder=5)

# Add labels to centroids with better styling
for i, centroid in enumerate(centroids_pca):
    plt.annotate(f'C{i}', xy=centroid, xytext=(0, 0),
                textcoords='offset points', fontsize=14,
                fontweight='bold', color='white',
                ha='center', va='center', zorder=6)

plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)',
           fontsize=13, fontweight='bold')
plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)',
           fontsize=13, fontweight='bold')
plt.title('Dengue Risk Clusters - PCA Visualization\n(K-Means Clustering with k=3, Risk Score: 1-100)',
          fontsize=15, fontweight='bold', pad=20)
plt.legend(loc='best', fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the plot
plt.savefig('dengue_risk_clusters_pca.png', dpi=300, bbox_inches='tight')
print("\nPCA plot saved as 'dengue_risk_clusters_pca.png'")



# ******************************************************
# 8. CLUSTER CHARACTERISTICS

print("\n[8] Analyzing Cluster Characteristics...")

cluster_summary = df.groupby('Cluster').agg({
    'Dengue_Cases': ['mean', 'min', 'max'],
    'Dengue_Deaths': ['mean', 'min', 'max'],
    'Temperature_C': ['mean', 'min', 'max'],
    'Humidity_pct': ['mean', 'min', 'max'], 
    'Rainfall_mm': ['mean', 'min', 'max'],
    'Risk_Score': ['mean', 'min', 'max']
}).round(2)

print("\nCluster Characteristics:")
print(cluster_summary)

# **********************************************
# 9. SAVE MODEL AND RESULTS


print("\n[9] Saving Model and Results...")

# Save the K-Means model and scaler
model_data = {
    'kmeans_model': kmeans,
    'scaler': scaler,
    'pca': pca,
    'feature_columns': feature_columns
}

joblib.dump(model_data, 'dengue_risk_model.pkl')
print("\nModel saved as 'dengue_risk_model.pkl'")

# Save results to CSV
output_columns = ['Month', 'Year', 'Region', 'Dengue_Cases', 'Dengue_Deaths',
                  'Temperature_C', 'Humidity_pct', 'Rainfall_mm',
                  'Cluster', 'Risk_Score', 'Risk_Category'] 

# Select only existing columns
output_columns = [col for col in output_columns if col in df.columns]

df_output = df[output_columns].copy()

df_output.to_csv('dengue_risk_predictions.csv', index=False)
print("Results saved as 'dengue_risk_predictions.csv'")

# ***********************************************************
# 10. SUMMARY REPORT


print("\n" + "=" * 60)
print("SUMMARY REPORT")
print("=" * 60)

print(f"\nTotal Records Analyzed: {len(df)}")
print(f"\nRisk Category Breakdown:")
for category in ['Low Risk (0-40)', 'Medium Risk (40-70)', 'High Risk (70-100)']:
    count = (df['Risk_Category'] == category).sum()
    percentage = (count / len(df)) * 100
    avg_score = df[df['Risk_Category'] == category]['Risk_Score'].mean()
    print(f"  - {category}: {count} ({percentage:.1f}%) | Avg Score: {avg_score:.2f}")


print(f"\nModel Performance Metrics:")
print(f"  - Silhouette Score: {silhouette_avg:.4f}")
print(f"  - Davies-Bouldin Index: {davies_bouldin:.4f}")
print(f"  - Calinski-Harabasz Score: {calinski_harabasz:.4f}")

print(f"\nPCA Variance Explained:")
print(f"  - PC1: {pca.explained_variance_ratio_[0]:.2%}")
print(f"  - PC2: {pca.explained_variance_ratio_[1]:.2%}")
print(f"  - Total: {sum(pca.explained_variance_ratio_):.2%}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE!")
print("=" * 60)

print("\nGenerated Files:")
print("  1. dengue_risk_model.pkl - Trained model with StandardScaler")
print("  2. dengue_risk_predictions.csv - Results with risk scores (1-100)")
print("  3. dengue_risk_clusters_pca.png - PCA cluster visualization")
print("  4. dengue_risk_analysis.png - Additional analysis plots")


# Show high risk regions
print("\nTop 10 Highest Risk Predictions:")
high_risk = df_output.nlargest(10, 'Risk_Score')[display_cols]
print(high_risk.to_string(index=False))

plt.show()

print("\n✓ All tasks completed successfully!")
print("✓ Risk Scores calculated on scale 1-100")
print("✓ StandardScaler used for feature scaling")
print("✓ Risk Categories: Low (0-40), Medium (40-70), High (70-100)")