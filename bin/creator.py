import os
import json
from loguru import logger
from checker import pathChecker
from torch.utils.data import DataLoader, random_split
from datalib import CustomImageDataset


# get file list each className
def getFileList(classInfoDatList, classInfo, imagePathInfo, phase='train'):

    for classInfoDat in classInfoDatList:
        if os.path.isfile(classInfoDat):
            dirPath = os.path.dirname(classInfoDat)

            with open(classInfoDat, "r") as f:
                classInfoData = json.load(f)

            classData = classInfoData["classData"]

            for _classData in classData:
                classNameCheck = (_classInfo for _classInfo in classInfo if _classInfo["className"] == _classData["className"])
                classObject = next(classNameCheck, False)

                if classObject:
                    filePathList = [os.path.join(dirPath, fileName) for fileName in _classData["filePath"]]
                    filePathList = pathChecker(filePathList, type="file")

                    className = _classData["className"]

                    if imagePathInfo.get(className):
                        imagePathInfo[className].extend(filePathList)
                        imagePathInfo[className] = list(set(imagePathInfo[className]))
                    else:
                        imagePathInfo[className] = filePathList
                        imagePathInfo[className] = list(set(imagePathInfo[className]))

                else:
                    classNameCheck = (_classInfo for _classInfo in classInfo if "sourceClassName" in _classInfo and _classInfo["sourceClassName"] == _classData["className"])
                    classObject = next(classNameCheck, False)

                    if classObject:
                        filePathList = [os.path.join(dirPath, fileName) for fileName in _classData["filePath"]]
                        filePathList = pathChecker(filePathList, type="file")
                        className = _classData["className"]

                        if imagePathInfo.get(className):
                            imagePathInfo[className].extend(filePathList)
                            imagePathInfo[className] = list(set(imagePathInfo[className]))
                        else:
                            imagePathInfo[className] = filePathList
                            imagePathInfo[className] = list(set(imagePathInfo[className]))

    return imagePathInfo


def createDataLoader(trainPathList, classInfo, classNameList, imageSize, batchSize, grayScale, trainRate):

    logger.info("createImageDataLoader")

    trainPathInfo = {}
    validPathInfo = {}
    imagePathInfo = {}
    classInfoDatList = []
    count = 0

    for trainPath in trainPathList:
        classInfoDatList.append(os.path.join(trainPath, "classInfo.dat"))
    imagePathInfo = getFileList(classInfoDatList, classInfo, imagePathInfo)

    for className in classNameList:
        count += len(imagePathInfo[className])

    if count == 0:
        raise Exception("Train ImagePath Length is 0, No Data!")

    if count > 50:
        logger.warning("Dataset size > 50")

        for className in classNameList:
            trainSize = int(len(imagePathInfo[className]) * float(trainRate["train"] / 100))
            validSize = len(imagePathInfo[className]) - trainSize

            trainSet, validSet = random_split(imagePathInfo[className], [trainSize, validSize])
            trainPathInfo[className] = trainSet
            validPathInfo[className] = validSet
            
    else:
        logger.warning("Dataset size < 50")

        for className in classNameList:
            trainSize = len(imagePathInfo[className])
            validSize = 0
       
            trainSet, validSet = random_split(imagePathInfo[className], [trainSize, validSize])
            trainPathInfo[className] = trainSet
            validPathInfo[className] = trainSet
    
    trainDataset = CustomImageDataset(trainPathInfo, classNameList, imageSize, grayScale)
    trainDataLoader = DataLoader(trainDataset, batch_size=batchSize, shuffle=True, num_workers=0)

    validDataset = CustomImageDataset(validPathInfo, classNameList, imageSize, grayScale)
    validDataLoader = DataLoader(validDataset, batch_size=batchSize, shuffle=True, num_workers=0)

    return trainDataLoader, validDataLoader
