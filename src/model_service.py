import lightgbm as lgb
import numpy as np
import glob 
import os

MODEL_FOLDER = "model/"



class ModelService:
    def __init__(self):
        filepath = glob.glob(os.path.join(MODEL_FOLDER, '*.txt'))
        self.models = [lgb.Booster(model_file=f) for f in filepath]

    def predict(self, features):
        features = np.array(features)
        
        # If it's a single sample (1D array), reshape to (1, n_features) for the model
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # model.predict() will return predictions for each sample in the batch
        preds = [model.predict(features) for model in self.models]
        
        # Average the predictions across all models for each sample
        # preds is a list of arrays: e.g., [array([p1_m1, p2_m1]), array([p1_m2, p2_m2])]
        # np.mean with axis=0 will compute: [ (p1_m1+p1_m2)/2, (p2_m1+p2_m2)/2 ]
        avg_preds = np.mean(preds, axis=0)
        
        return avg_preds
    
    def prepare_data(self, data):
        pass


# ---- Ví dụ dùng thử ----
if __name__ == "__main__":
    service = ModelService()
    # Example with a single prediction. 
    # The input now requires 21 features based on the latest logic in app.py.
    print("Single prediction:")
    # The result for a single prediction will be a numpy array with one element, e.g., [0.123]
    print(service.predict([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]))

    print("-" * 20)

    # Example with a batch of 2 predictions
    batch_data = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21],
        [21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    ]
    print("Batch prediction:")
    # The result for a batch prediction will be a numpy array with multiple elements, e.g., [0.123, 0.456]
    print(service.predict(batch_data))

