import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans as SklearnKMeans

class SimpleKMeans:
    def __init__(self, n_clusters=10, max_iter=300, tol=1e-4, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.cluster_centers_ = X[random_indices]

        for i in range(self.max_iter):
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)

            new_centers = np.array([
                X[labels == k].mean(axis=0) if np.sum(labels == k) > 0 else self.cluster_centers_[k]
                for k in range(self.n_clusters)
            ])

            center_shift = np.linalg.norm(self.cluster_centers_ - new_centers)
            self.cluster_centers_ = new_centers
            
            if center_shift < self.tol:
                break
                
        return self

    def predict(self, X):
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)

mnist = tf.keras.datasets.mnist
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

subset_size = 10000 
X_train = X_train[:subset_size]
X_test = X_test[:2000]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

sklearn_km = SklearnKMeans(n_clusters=10, random_state=42, n_init=10, algorithm='lloyd')
sklearn_km.fit(X_train_scaled)
sklearn_preds = sklearn_km.predict(X_test_scaled)
sklearn_centers = sklearn_km.cluster_centers_

custom_km = SimpleKMeans(n_clusters=10, random_state=42)
custom_km.fit(X_train_scaled)
custom_preds = custom_km.predict(X_test_scaled)
custom_centers = custom_km.cluster_centers_


X_combined = np.vstack([X_test_scaled, sklearn_centers, custom_centers])
tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne_combined = tsne.fit_transform(X_combined)

X_test_tsne = X_tsne_combined[:len(X_test_scaled)]
sk_centers_tsne = X_tsne_combined[len(X_test_scaled):len(X_test_scaled)+10]
cu_centers_tsne = X_tsne_combined[len(X_test_scaled)+10:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

ax1.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=sklearn_preds, cmap='tab10', alpha=0.5, s=10)
ax1.scatter(
    sk_centers_tsne[:, 0],
    sk_centers_tsne[:, 1],
    c='red',
    marker='X',
    s=200,
    edgecolors='black',
    label='Centra Sklearn'
)
ax1.set_title("Biblioteka Sklearn KMeans")
ax1.legend()

ax2.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=custom_preds, cmap='tab10', alpha=0.5, s=10)
ax2.scatter(
    cu_centers_tsne[:, 0],
    cu_centers_tsne[:, 1],
    c='red',
    marker='X',
    s=200,
    edgecolors='black',
    label='Centra Własne'
)
ax2.set_title("Własna implementacja (SimpleKMeans)")
ax2.legend()

plt.suptitle(f"Porównanie klastrów")
plt.show()
