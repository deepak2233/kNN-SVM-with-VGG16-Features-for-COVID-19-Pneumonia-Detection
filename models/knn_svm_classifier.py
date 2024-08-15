from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

def knn_svm_classifier(X_train, X_test, y_train, y_test, k=5, c=1.0):
    # kNN-SVM model (SVM combined with kNN)
    knn = KNeighborsClassifier(n_neighbors=k)
    svm = SVC(kernel='linear', C=c)
    
    # kNN-SVM Pipeline
    model = make_pipeline(StandardScaler(), knn, svm)
    model.fit(X_train, y_train)
    
    # Prediction and evaluation
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
