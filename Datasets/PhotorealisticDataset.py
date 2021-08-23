"""

    Created on 23/08/21 8:44 AM 
    @author: Kartik Prabhu

"""

from PIL import Image
from torch.utils.data import Dataset
import os

class PhotoRealDataset(Dataset):

    def __init__(self, root_path,images_per_cat=10, transforms=None):
        self.image_paths = []
        self.labels = []
        self.transform = transforms
        self.size = [224,224]

        for folder in os.listdir(root_path):
            count = 0
            if(folder !='.DS_Store'):
                for file in os.listdir(os.path.join(root_path,folder)):
                    # instances.append(cv2.imread(os.path.join(os.path.join(root_path,folder),file),0))
                    if file == ".DS_Store":
                        continue

                    count += 1
                    self.image_paths.append(os.path.join(os.path.join(root_path, folder), file))
                    self.labels.append(folder)

                    if count == images_per_cat:
                        break

    def __getitem__(self, idx):

        input_img = self.read_img(self.image_paths[idx])
        input_img = self.transform(input_img)

        return input_img, self.labels[idx]


    def read_img(self, path, type='RGB'):
        img =Image.open(path).convert("RGB")
        return img

    def __len__(self):
        return len(self.image_paths)