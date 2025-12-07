import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn. manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = load_wine()
X = data.data
y = data.target
df = pd.DataFrame(X, columns=data.feature_names)
print(df.head())

# Train: 60%, Val: 20%, Test: 20%
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.40, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=3, random_state=42)
train_clusters = kmeans.fit_predict(X_train_scaled)

cluster_to_class = {}
for cluster_id in np.unique(train_clusters):
    mask = train_clusters == cluster_id
    most_common_class = np.bincount(y_train[mask]).argmax()
    cluster_to_class[cluster_id] = most_common_class

# train
y_train_pred = np.array([cluster_to_class[c] for c in train_clusters])
# val
val_clusters = kmeans.predict(X_val_scaled)
y_val_pred = np.array([cluster_to_class[c] for c in val_clusters])
# test
test_clusters = kmeans.predict(X_test_scaled)
y_test_pred = np.array([cluster_to_class[c] for c in test_clusters])
# classification accuracy
print("Train accuracy:", accuracy_score(y_train, y_train_pred))
print("Val accuracy: ", accuracy_score(y_val, y_val_pred))
print("Test accuracy: ", accuracy_score(y_test, y_test_pred))

X_all_scaled = scaler.transform(X) # UWAGA: transform, nie fit_transform!
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_all_scaled)
plt.figure(figsize=(7,5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Wizualizacja prawdziwych klas po t-SNE (Wine Dataset)")
plt.show()

all_clusters = kmeans.predict(X_all_scaled)
plt.figure(figsize=(7,5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=all_clusters)
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Wizualizacja klastrów k-means po t-SNE (Wine Dataset)")
plt.show()

# załóżmy, że all_clusters to przewidywania k-means
for cluster_id in np.unique(all_clusters):
    plt.scatter(
        X_tsne[all_clusters == cluster_id, 0],
        X_tsne[all_clusters == cluster_id, 1],
        label=f'Klaster {cluster_id}'
    )
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Wizualizacja klastrów k-means po t-SNE (Wine Dataset)")
plt.legend()
plt.show()
