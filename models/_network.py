from abc import ABC, abstractmethod
from skimage import io
import torch

class Network:
    @abstractmethod
    def __init__(self, weigths_path, df_authors):
        self.model = None
        self.df_authors = df_authors
    
    def predict(self, image_path):
        image = io.imread(image_path)
        image = self.prepare_image(image)
        outputs = self.model(image)
        _, author_id = torch.max(outputs, 1)
        author_id = author_id.cpu().detach().numpy()[0]
        return self.df_authors.iloc[author_id]['name']
    
    @abstractmethod
    def prepare_image(self, image):
        pass
