import joblib

def load_model(model_path='models/svm.pkl'):
    return joblib.load(model_path)

def predict(model, X):
    return model.predict(X)
