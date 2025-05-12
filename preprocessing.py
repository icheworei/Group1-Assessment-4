import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
import numpy as np

# Load dataset
df = pd.read_excel("AvianData.xlsx")

# Extract Year and Month
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df = df.drop(columns=['Date'])

# Create a binary variable: Outbreak (Yes/No)
df['Outbreak'] = df['Colisepticaemia Cases'].apply(lambda x: 'Yes' if x >= 2 else 'No')
df = df.drop(columns=['Colisepticaemia Cases'])

# Handle missing data
print("Missing values per column:\n", df.isnull().sum())

# Specify the feature variables and the target variable.
X = df.drop(columns=['Outbreak'])
y = df['Outbreak']

# Separate categorical and numerical columns
categorical_cols = ['Region', 'Age Category']  # Excluding 'Farm Type' and 'Migration Season'
numerical_cols = ['Temperature (°C)', 'Rainfall (mm)', 'Sunshine', 'Humidity (%)', 'Year', 'Month']

# Preprocess data
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols),
        ('num', StandardScaler(), numerical_cols)
    ])

X_processed = preprocessor.fit_transform(X)

# Perform Principal Component Analysis (PCA) on the numerical variables.
pca = PCA(n_components=0.95)
numerical_data = X_processed[:, len(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols)):]
pca_result = pca.fit_transform(numerical_data)
n_components = pca.n_components_
print(f"Number of components explaining 95% variance: {n_components}")

# Combine PCA results with categorical data
categorical_data = X_processed[:, :len(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_cols))]
X_pca = np.hstack((categorical_data, pca_result))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Save preprocessed data
pd.DataFrame(X_pca).to_csv("preprocessed_data_pca.csv", index=False)
print("Preprocessing complete. Training set size:", X_train.shape, "Test set size:", X_test.shape)
