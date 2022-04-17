from ._network import Network
from torchvision import transforms
import torch
import pandas as pd
from skimage.color import gray2rgb
from lib.SReT import SReT_S


class SReT(Network):
    def __init__(self, weigths_path, df_authors_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = SReT_S(pretrained=False)
        self.model.head = torch.nn.Linear(504, 30)
        self.model.load_state_dict(torch.load(weigths_path))
        self.model.eval()
        self.model.to(self.device)
        
        self.df_authors = pd.read_csv(df_authors_path)
    
    def prepare_image(self, image):
        transforms_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.Resize(224),
            transforms.RandomCrop(224)
        ])
        
        if len(image.shape) == 2 or image.shape[2] == 1:
            image = gray2rgb(image)

        image = transforms_test(image).unsqueeze(0).to(self.device)
        return image
