import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

class KNNSVMClassifier:
    def __init__(self, k=5, kernel='linear'):
        self.k = k
        self.kernel = kernel
        self.scaler = StandardScaler()
        self.svm = SVC(kernel=kernel, class_weight='balanced')

    def calculate_knn_weights(self, X_train, X_test):
        knn = NearestNeighbors(n_neighbors=self.k)
        knn.fit(X_train)
        distances, indices = knn.kneighbors(X_test)
        weights = np.exp(-distances)
        return weights.mean(axis=1)

    def fit(self, X_train, y_train):
        X_train_scaled = self.scaler.fit_transform(X_train)
        train_weights = self.calculate_knn_weights(X_train_scaled, X_train_scaled)
        self.svm.fit(X_train_scaled, y_train, sample_weight=train_weights)

    def predict(self, X_test):
        X_test_scaled = self.scaler.transform(X_test)
        return self.svm.predict(X_test_scaled)
