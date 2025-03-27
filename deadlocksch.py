import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

class AIDeadlockDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)

    def train(self, X, y):
        """ Train the model on historical deadlock cases """
        self.model.fit(X, y)
        joblib.dump(self.model, "models/deadlock_model.pkl")

    def predict(self, allocation_matrix, request_matrix):
        """ Predict whether a deadlock will occur """
        model = joblib.load("models/deadlock_model.pkl")
        X_input = np.hstack((allocation_matrix.flatten(), request_matrix.flatten())).reshape(1, -1)
        prediction = model.predict(X_input)
        return "Deadlock Detected!" if prediction[0] == 1 else "Safe State"

detector = AIDeadlockDetector()

X_train = np.array([
    [1, 0, 1, 0, 0, 1, 1, 1, 0],  
    [2, 1, 3, 1, 0, 2, 0, 1, 0],  
    [0, 2, 1, 0, 1, 0, 1, 0, 0],  
])
y_train = np.array([1, 0, 1])  

detector.train(X_train, y_train)

allocation = np.array([[1, 0, 1], [2, 1, 3], [0, 2, 1]])
request = np.array([[0, 0, 1], [1, 0, 2], [0, 1, 0]])

print(detector.predict(allocation, request))
