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
        self.unique_label = []
        for folder in os.listdir(root_path):
            count = 0
            # print(folder)
            if(folder !='.DS_Store'):
                if not (folder in self.unique_label):
                    self.unique_label.append(folder)
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

    def get_unique_labels(self):
        return self.unique_label

    def read_img(self, path, type='RGB'):
        img =Image.open(path).convert("RGB")
        return img

    def __len__(self):
        return len(self.image_paths)


class PhotoRealDataset2(Dataset):

    def __init__(self, root_path,images_per_cat=10, transforms=None):
        self.images_per_cat=images_per_cat
        self.image_paths = []
        self.labels = []
        self.transform = transforms
        self.size = [224,224]
        self.unique_label = []
        self.pix3d = []

        self.dataset = []

        for folder in os.listdir(root_path):
            count = 0
            # print(folder)

            # if folder == 'pix3d':
            #     continue

            if(folder !='.DS_Store'):

                if not (folder in self.unique_label):
                    self.unique_label.append(folder)
                for file in os.listdir(os.path.join(root_path,folder)):
                    # instances.append(cv2.imread(os.path.join(os.path.join(root_path,folder),file),0))
                    if file == ".DS_Store":
                        continue

                    self.dataset.append(folder)


                    count += 1
                    self.image_paths.append(os.path.join(os.path.join(root_path, folder), file))
                    self.labels.append(folder)

                    if count == images_per_cat:
                        break

        count = 0
        for file in os.listdir(os.path.join(root_path,'pix3d')):
            if file == ".DS_Store":
                continue

            count += 1
            self.pix3d.append(os.path.join(os.path.join(root_path, 'pix3d'), file))

            if count == images_per_cat:
                break


        print(len(self.image_paths))
        print(len(self.dataset))


    def __getitem__(self, idx):
        input_img = self.read_img(self.image_paths[idx])
        input_img = self.transform(input_img)

        pix3d_img = self.read_img(self.pix3d[idx%self.images_per_cat])
        pix3d_img = self.transform(pix3d_img)

        return self.dataset[idx],input_img, pix3d_img

    def get_unique_labels(self):
        return self.unique_label

    def read_img(self, path, type='RGB'):
        img =Image.open(path).convert("RGB")
        return img

    def __len__(self):
        return len(self.image_paths)