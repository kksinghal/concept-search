import torch
from models import *
import os
import numpy as np

np.random.seed(10)
torch.manual_seed(10)

class TransformationToVec:
    def __init__(self, model_dir) -> None:
        self.models = []
        for model_name in os.listdir(model_dir):
            model_path = os.path.join(model_dir, model_name)
            model = Model()
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
            self.models.append(model)

    def get(self, inputs, outputs):
        X = []
        Y = []
        # One-hot encoding of colors into 10 channels 
        for i, (x, y) in enumerate(zip(inputs, outputs)):
            u = np.arange(10) 
            x = (u[:,np.newaxis,np.newaxis]==x).astype(int)
            y = (u[:,np.newaxis,np.newaxis]==y).astype(int)

            X.append(torch.tensor(x, dtype=torch.float64))
            Y.append(torch.tensor(y, dtype=torch.float64))
            
        vec = None
        for model in self.models:
            with torch.no_grad():
                if vec is None:
                    vec = model.feature_extractor([X], [Y])[0]
                else:
                    vec += model.feature_extractor([X], [Y])[0]

        return vec / len(self.models)