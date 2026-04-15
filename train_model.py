import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

df.drop(columns=['id'], inplace=True)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

# Encode categorical
le_dict = {}
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

X = df.drop('stroke', axis=1)
y = df['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA for feature reduction
pca = PCA(n_components=8)
X_reduced = pca.fit_transform(X_scaled)

# Random Forest for better accuracy
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_reduced, y)

# Save
pickle.dump(scaler, open('scaler.pkl','wb'))
pickle.dump(rf, open('mlp.pkl','wb'))  # reusing mlp.pkl name
pickle.dump(pca, open('pca.pkl','wb'))

print("Training complete!")