from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np

class PolygonDataset:
    def __init__(self, shape_types, num_samples=200, image_size=28):
        self.shape_types = shape_types
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = []
        self.labels = []
        self.generate_data()

    def generate_data(self):
        for label, shape in enumerate(self.shape_types):
            for _ in range(self.num_samples):
                img = Image.new("L", (self.image_size, self.image_size), 0)
                draw = ImageDraw.Draw(img)
                s = self.image_size // 4
                x, y = self.image_size//2, self.image_size//2
                if shape == "circle":
                    draw.ellipse([x-s, y-s, x+s, y+s], fill=255)
                elif shape == "triangle":
                    draw.polygon([(x, y-s), (x-s, y+s), (x+s, y+s)], fill=255)
                elif shape == "square":
                    draw.rectangle([x-s, y-s, x+s, y+s], fill=255)
                # add more shapes here
                self.data.append(np.array(img)/255.0)
                self.labels.append(label)