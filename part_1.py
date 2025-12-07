import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

mnist = tf.keras.datasets.mnist
(X_train_raw, y_train), (X_test_raw, y_test) = mnist.load_data()

X_train = X_train_raw.reshape(X_train_raw.shape[0], -1)
X_test = X_test_raw.reshape(X_test_raw.shape[0], -1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

test_clusters = kmeans.predict(X_test_scaled)
centers = kmeans.cluster_centers_

X_combined = np.vstack([X_test_scaled, centers])

tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
X_tsne_combined = tsne.fit_transform(X_combined)

X_test_tsne = X_tsne_combined[:-10]
centers_tsne = X_tsne_combined[-10:]

plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    X_test_tsne[:, 0], 
    X_test_tsne[:, 1], 
    c=test_clusters, 
    cmap='tab10', 
    alpha=0.6, 
    s=10,
    label='Dane testowe'
)

plt.scatter(
    centers_tsne[:, 0],
    centers_tsne[:, 1],
    c='red',
    marker='X',
    s=200,
    edgecolors='black', 
    linewidth=2,
    label='Środki klastrów (Centers)'
)

plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.title("Wizualizacja klastrów K-Means na zbiorze testowym MNIST")
plt.legend()
plt.colorbar(scatter, label='ID Klastra')
plt.show()
