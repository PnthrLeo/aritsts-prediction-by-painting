from ._network import Network
from torchvision import transforms
import torch
import pandas as pd
from skimage.color import gray2rgb


class EfficientNet(Network):
    def __init__(self, weigths_path, df_authors_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_efficientnet_b4', pretrained=False)
        self.model.classifier.fc = torch.nn.Linear(1792, 30)
        self.model.load_state_dict(torch.load(weigths_path))
        self.model.eval()
        self.model.to(self.device)
        
        self.df_authors = pd.read_csv(df_authors_path)
    
    def prepare_image(self, image):
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(380),
            transforms.RandomCrop(380)
        ])
        
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        image = transforms_test(image).unsqueeze(0).to(self.device)
        return image
    