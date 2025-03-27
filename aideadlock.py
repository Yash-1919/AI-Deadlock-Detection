import numpy as np
from sklearn.ensemble import RandomForestClassifier

class AIDeadlockDetector:
    def __init__(self, processes, resources, allocation, request, history_data):
        self.processes = processes
        self.resources = resources
        self.allocation = np.array(allocation)
        self.request = np.array(request)
        self.history_data = history_data
        self.model = RandomForestClassifier(n_estimators=100)
    
    def train_model(self):
        X, y = self.history_data['features'], self.history_data['labels']
        self.model.fit(X, y)
    
    def predict_deadlock(self):
        current_state = np.hstack((self.allocation.flatten(), self.request.flatten()))
        prediction = self.model.predict([current_state])
        return 'Deadlock Likely' if prediction[0] == 1 else 'No Deadlock'
    
    
history_data = {
    'features': [[0, 1, 0, 2, 0, 0], [3, 0, 2, 2, 1, 1]],
    'labels': [0, 1]  
}
processes = ['P1', 'P2', 'P3', 'P4', 'P5']
resources = [10, 5, 7]
allocation = [[0, 1, 0], [2, 0, 0], [3, 0, 2], [2, 1, 1], [0, 0, 2]]
request = [[0, 0, 0], [2, 0, 2], [0, 0, 0], [1, 0, 0], [0, 0, 2]]

detector = AIDeadlockDetector(processes, resources, allocation, request, history_data)
detector.train_model()
print(detector.predict_deadlock())
