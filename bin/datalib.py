import cv2
import torch
from torchvision import transforms
from PIL import Image


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, imgPathInfo, classNameList, imageSize, grayScale):
        self.imgPathInfo = imgPathInfo
        self.classNameList = classNameList
        self.labelList = []
        self.imagePathList = []
        self.grayScale = grayScale

        if self.grayScale == 1:
            self.transform = transforms.Compose([
                transforms.Resize((imageSize, imageSize)),
                transforms.Grayscale(num_output_channels=int(grayScale)),
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.5),
            ])

        else:
            self.transform = transforms.Compose([
                transforms.Resize((imageSize, imageSize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        for key, value in self.imgPathInfo.items():
            for filePath in value:
                self.labelList.append([self.classNameList.index(key)])
                self.imagePathList.append(filePath)

    def __len__(self):
        return len(self.labelList)

    def __getitem__(self, idx):
        imgPath = self.imagePathList[idx]
        label = self.labelList[idx]
       
        image = cv2.imread(imgPath)

        if self.grayScale == 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = Image.fromarray(image)

        image = self.transform(image)
        label = torch.LongTensor(label)[0]
        
        return image, label