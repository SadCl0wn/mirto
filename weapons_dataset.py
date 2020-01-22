from torchvision.datasets import VisionDataset
from bs4 import BeautifulSoup
from PIL import Image
import os
import os.path
import sys

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class WeaponS(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(WeaponS, self).__init__(root, transform=transform, target_transform=target_transform)
        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')
        self._data = []
        self._class = {}
        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        dataFiles = filter(lambda name: name.endswith(".jpg"),os.listdir("WeaponS"))
        for path_img in dataFiles:
            class_name = path_img.split('/')[0]
            #self._class[class_name] += 1 #assume ordered fill if not need to be a list of indexs
            self._class[class_name].append(len(self._data)) #list of indexs of entities for fair split
            target = {
                "boxes":[],
                "labels":[],
                "image_id":len(self._data),
                "area":[],
                "iscrowd":[],
            }
            with open("{0}/{1}.xml".format("WeaponS_bbox",path_img)) as f:
                xml = BeautifulSoup(f.read(),"xml")
                for o in xml.object:
                    target["labels"].append(o.name)
                    target["boxes"].append([o.bndbox.xmin,o.bndbox.ymin,o.bndbox.xmax,o.bndbox.ymax])
                    target["area"].append((o.bndbox.xmax-o.bndbox.xmin)*(o.bndbox.ymax-o.bndbox.ymin))
                    target["iscrowd"].append(0)
            self._data.append(["{0}/{1}.jpg".format("WeaponS",path_img),target])

    def getPopulationIndexs(self,target):
        return self._class[target]

    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''

        image_path, target = self._data[index] # Provide a way to access image and target via index
                           # Image should be a PIL Image
                           # target is a dict
        # Applies preprocessing when accessing the image
        image = pil_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self._data) # Provide a way to get the length (number of elements) of the dataset
        return length