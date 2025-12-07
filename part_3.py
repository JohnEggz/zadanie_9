import sys
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QVBoxLayout,
    QHBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
)
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class StepKMeans:
    def __init__(self, n_clusters=10, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.centers = None
        self.labels = None
        self.X_high_dim = None
        self.iteration = 0
        
    def initialize(self, X):
        self.X_high_dim = X
        self.iteration = 0
        np.random.seed(self.random_state)
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        self.centers = X[indices]
        self._assign_labels()

    def _assign_labels(self):
        distances = np.linalg.norm(self.X_high_dim[:, np.newaxis] - self.centers, axis=2)
        self.labels = np.argmin(distances, axis=1)

    def step(self):
        new_centers = np.array([
            self.X_high_dim[self.labels == k].mean(axis=0) 
            if np.sum(self.labels == k) > 0 else self.centers[k]
            for k in range(self.n_clusters)
        ])
        
        diff = np.linalg.norm(self.centers - new_centers)
        self.centers = new_centers
        self._assign_labels()
        self.iteration += 1
        
        return diff

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)

class KMeansWindow(QMainWindow):
    def __init__(self, X_high, X_tsne):
        super().__init__()
        self.setWindowTitle("Wizualizacja treningu K-Means (t-SNE)")
        self.resize(1000, 800)

        self.X_high = X_high
        self.X_tsne = X_tsne
        self.kmeans = StepKMeans(n_clusters=10)
        self.kmeans.initialize(self.X_high)

        self.timer = QTimer()
        self.timer.setInterval(200)
        self.timer.timeout.connect(self.run_step)

        self._init_ui()
        self.update_plot()

    def _init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        layout.addWidget(self.canvas)

        controls_layout = QHBoxLayout()
        
        self.iter_label = QLabel("Iteracja: 0")
        self.iter_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        controls_layout.addWidget(self.iter_label)

        self.btn_step = QPushButton("Krok +1")
        self.btn_step.clicked.connect(self.run_step)
        controls_layout.addWidget(self.btn_step)

        self.btn_play = QPushButton("Auto Start")
        self.btn_play.setCheckable(True)
        self.btn_play.clicked.connect(self.toggle_animation)
        controls_layout.addWidget(self.btn_play)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_model)
        controls_layout.addWidget(self.btn_reset)

        layout.addLayout(controls_layout)

    def reset_model(self):
        self.timer.stop()
        self.btn_play.setChecked(False)
        self.btn_play.setText("Auto Start")
        self.kmeans.random_state += 1 
        self.kmeans.initialize(self.X_high)
        self.update_plot()

    def toggle_animation(self):
        if self.btn_play.isChecked():
            self.btn_play.setText("Stop")
            self.timer.start()
        else:
            self.btn_play.setText("Auto Start")
            self.timer.stop()

    def run_step(self):
        diff = self.kmeans.step()
        self.update_plot()
        if diff < 1e-4:
            self.timer.stop()
            self.btn_play.setChecked(False)
            self.btn_play.setText("Koniec (Zbieżność)")

    def update_plot(self):
        self.iter_label.setText(f"Iteracja: {self.kmeans.iteration}")
        ax = self.canvas.axes
        ax.clear()

        scatter = ax.scatter(
            self.X_tsne[:, 0], 
            self.X_tsne[:, 1], 
            c=self.kmeans.labels, 
            cmap='tab10', 
            s=10, 
            alpha=0.6
        )

        visual_centers = []
        for k in range(self.kmeans.n_clusters):
            mask = (self.kmeans.labels == k)
            if np.sum(mask) > 0:
                visual_center = self.X_tsne[mask].mean(axis=0)
                visual_centers.append(visual_center)
        
        if visual_centers:
            visual_centers = np.array(visual_centers)
            ax.scatter(
                visual_centers[:, 0], 
                visual_centers[:, 1], 
                c='white', 
                s=200, 
                marker='X', 
                edgecolors='black', 
                linewidth=2,
                label='Środki (Wizualne)'
            )

        ax.set_title(f"Wizualizacja K-Means na mapie t-SNE (Iteracja {self.kmeans.iteration})")
        # ax.axis('off')
        self.canvas.draw()


if __name__ == "__main__":
    mnist = tf.keras.datasets.mnist
    (X_train_raw, _), (_, _) = mnist.load_data()

    subset_size = 1500
    X_subset = X_train_raw[:subset_size].reshape(subset_size, -1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_subset)
    
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=42)
    X_tsne_2d = tsne.fit_transform(X_scaled)

    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = KMeansWindow(X_scaled, X_tsne_2d)
    window.show()
    sys.exit(app.exec_())
