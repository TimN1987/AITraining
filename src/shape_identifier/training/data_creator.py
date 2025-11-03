from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import numpy as np
import random
import math

class PolygonDataset:
    def __init__(self, shape_types, num_samples=200, image_size=28):
        self.shape_types = shape_types
        self.num_samples = num_samples
        self.image_size = image_size
        self.data = []
        self.labels = []
        self.generate_data()

    def random_transform(self, img):
        """Apply random rotation, translation, and scaling."""
        # random rotation (-30° to +30°)
        angle = random.uniform(-30, 30)
        img = img.rotate(angle, fillcolor=0)

        # random translation (up to 20% of image size)
        max_shift = int(self.image_size * 0.2)
        dx = random.randint(-max_shift, max_shift)
        dy = random.randint(-max_shift, max_shift)
        img = img.transform(
            img.size,
            Image.AFFINE,
            (1, 0, dx, 0, 1, dy),
            fillcolor=0
        )

        # random scaling (zoom in/out)
        scale = random.uniform(0.8, 1.2)
        new_size = int(self.image_size * scale)
        img_resized = img.resize((new_size, new_size))
        new_img = Image.new("L", (self.image_size, self.image_size), 0)
        paste_x = (self.image_size - new_size) // 2
        paste_y = (self.image_size - new_size) // 2
        new_img.paste(img_resized, (paste_x, paste_y))
        return new_img

    def generate_data(self):
        for label, shape in enumerate(self.shape_types):
            for _ in range(self.num_samples):
                img = Image.new("L", (self.image_size, self.image_size), 0)
                draw = ImageDraw.Draw(img)

                # base shape parameters
                s = random.randint(self.image_size // 5, self.image_size // 3)
                x, y = self.image_size // 2, self.image_size // 2

                # draw the shape
                if shape == "circle":
                    draw.ellipse([x - s, y - s, x + s, y + s], fill=255)
                elif shape == "triangle":
                    draw.polygon([(x, y - s), (x - s, y + s), (x + s, y + s)], fill=255)
                elif shape == "square":
                    draw.rectangle([x - s, y - s, x + s, y + s], fill=255)
                elif shape == "rectangle":
                    draw.rectangle([x - s, y - s//2, x + s, y + s//2], fill=255)
                elif shape == "pentagon":    
                    points = [
                        (x + s * math.cos(2*math.pi*i/5), y + s * math.sin(2*math.pi*i/5)) for i in range(5)
                    ]
                    draw.polygon(points, fill=255)
                elif shape == "hexagon":
                    points = [
                        (x + s * math.cos(2*math.pi*i/6), y + s * math.sin(2*math.pi*i/6)) for i in range(6)
                    ]
                    draw.polygon(points, fill=255)
                elif shape == "diamond":
                    draw.polygon([(x, y - s), (x - s, y), (x, y + s), (x + s, y)], fill=255)
                elif shape == "star":
                    num_points = random.randint(5, 12)
                    outer_radius = s
                    inner_radius = s / 2
                    points = []
                    angle_step = 360 / (num_points * 2)
                    for i in range(num_points * 2):
                        r = outer_radius if i % 2 == 0 else inner_radius
                        angle = math.radians(i * angle_step)
                        points.append((x + r * math.cos(angle), y + r * math.sin(angle)))
                    draw.polygon(points, fill=255)

                # apply random augmentation
                img = self.random_transform(img)

                # store
                self.data.append(np.array(img) / 255.0)
                self.labels.append(label)
