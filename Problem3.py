import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D


#import the breast _cancer dataset
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()
print(data.keys())

# Check the output classes
print(data['target_names'])

# Check the input attributes
print(data['feature_names'])

# construct a dataframe using pandas
df1=pd.DataFrame(data['data'],columns=data['feature_names'])

# Scale data before applying PCA
scaling=StandardScaler()

# Use fit and transform method
scaled_df = scaling.fit_transform(df1)

# Set the n_components=3
principal=PCA(n_components=3)
x = principal.fit_transform(scaled_df)

# Check the dimensions of data after PCA
print(x.shape)

# Columns forming the principal components
print(principal.components_)


# Plot the principal components
# 2D Plot
plt.figure(figsize=(10,10))
plt.scatter(x[:,0],x[:,1],c=data['target'],cmap='plasma')
plt.xlabel('pc1')
plt.ylabel('pc2')
plt.show()

# 3D Plot
fig = plt.figure(figsize=(10,10))
axis = fig.add_subplot(111, projection='3d')
axis.scatter(x[:,0],x[:,1],x[:,2], c=data['target'],cmap='plasma')
axis.set_xlabel("PC1", fontsize=10)
axis.set_ylabel("PC2", fontsize=10)
axis.set_zlabel("PC3", fontsize=10)

plt.show()

# Calculate variance ratio
print(principal.explained_variance_ratio_)
