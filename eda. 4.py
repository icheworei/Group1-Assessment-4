import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np

# Load the original dataset
df = pd.read_excel("AvianData.xlsx")

# Extract Year and Month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df = df.drop(columns=['Date'])

# Create target variable: Outbreak (Yes/No)
df['Outbreak'] = df['Colisepticaemia Cases'].apply(lambda x: 'Yes' if x >= 2 else 'No')
df = df.drop(columns=['Colisepticaemia Cases'])

# Define features
X = df.drop(columns=['Outbreak'])

# Separate categorical and numerical columns
categorical_cols = ['Region', 'Age Category']
numerical_cols = ['Temperature (°C)', 'Rainfall (mm)', 'Sunshine', 'Humidity (%)', 'Year', 'Month']

# Preprocess data to get pca_result
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

X_processed = preprocessor.fit_transform(X)

# Apply PCA to numerical features
pca = PCA(n_components=0.95)
numerical_data = X_processed[:, len(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)):]
pca_result = pca.fit_transform(numerical_data)

# Class distribution
sns.countplot(x='Outbreak', data=df)
plt.title('Outbreak Distribution')
plt.show()

# Scatter plot of first two PCs
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=df['Outbreak'].map({'Yes': 1, 'No': 0}), cmap='viridis')
plt.title('PCA Components 1 vs 2')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.show()

# Correlation heatmap (before PCA)
df['Outbreak_binary'] = df['Outbreak'].map({'Yes': 1, 'No': 0})
correlation_matrix = df[['Temperature (°C)', 'Rainfall (mm)', 'Sunshine', 'Humidity (%)', 'Outbreak_binary']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Before PCA)')
plt.show()