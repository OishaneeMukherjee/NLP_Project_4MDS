from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import joblib

def train_svm(X, y, save_path='models/svm.pkl'):
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = SVC(probability=True)
    clf.fit(X_train, y_train)
    joblib.dump(clf, save_path)
    return clf, X_test, y_test
