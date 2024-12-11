import torch

torch.manual_seed(10)

N_CLASSES=16

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(10, 32, kernel_size=3, padding=1)
        # self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=1)

        self.dil_conv1 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)
        # self.dil_conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, padding=2, dilation=2)

        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.relu = torch.nn.ReLU()

    def extract_features(self, x):
        x = x.type(torch.float)

        o1 = self.conv1(x)
        o2 = self.relu(o1)
        # o2 = self.conv2(o1)
        # o2 = self.relu(o2)

        o3 = self.dil_conv1(o2) + o2
        o4 = self.relu(o3)

        # o4 = self.dil_conv2(o3) + o3
        # o4 = self.relu(o4)

        o5 = self.conv3(o4)
        o5 = self.relu(o5)

        o6 = torch.cat([torch.mean(o5, dim=[1,2], keepdim=False), torch.amax(o5, dim=[1,2], keepdim=False), torch.amin(o5, dim=[1,2], keepdim=False)]) # (c, w, h) -> (2c)
        return o6

    def forward(self, X, Y):
        N = len(X)

        batch_output = []
        for i in range(N):
            sample_out = None
            for x, y in zip(X[i], Y[i]):
                if sample_out is None: sample_out = self.extract_features(x) - self.extract_features(y)
                else: sample_out += self.extract_features(x) - self.extract_features(y)

            sample_out = sample_out / (len(X[i]))
            
            batch_output.append(sample_out)

        return torch.stack(batch_output)
        
class Projection(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(192, 128)

    def forward(self, X):
        return self.linear(X)
    

class Classifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(192, N_CLASSES)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        return self.linear1(X)
    

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.projection = Projection()
        self.classifier = Classifier()

    def forward(self, X, Y):
        features = self.feature_extractor(X, Y)
        projection = self.projection(features)
        logits = self.classifier(features)

        return projection, logits