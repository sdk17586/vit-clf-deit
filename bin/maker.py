import os
import sys
import json
import time
import traceback
from loguru import logger
from PIL import Image, ImageDraw
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.append(basePath)


# 디렉토리 중복 체크 및 생성
def makeDir(path):

    '''
        정보 1. 디렉토리에 폴더가 있는지 확인 및 생성
    '''

    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def makeMaskImage(imgPathList):
  
    '''
        정보 1. Segmentation 포인트 좌표를 마스크 이미지로 변환하는 로직
    '''

    try:
        logger.info("Making Mask of Test Data...!")
        makeMaskTime = time.time()

        for imagePath in imgPathList:
            img = Image.open(imagePath)

            root, file = os.path.split(imagePath)
            fileName, fileExtension = os.path.splitext(file)
            datPath = os.path.join(root, fileName + ".dat")

            maskPath = os.path.join(root, "MASK_" + fileName + "{}".format(fileExtension))
            mask = Image.new("RGB", img.size)

            with open(datPath, "r") as jsonFile:
                datJson = json.load(jsonFile)

            polygonData = datJson["polygonData"]
            brushData = datJson["brushData"]

            if len(polygonData) > 0:
                for i, pdata in enumerate(polygonData):
                    positionData = []
                    draw = ImageDraw.Draw(mask)

                    for i, position in enumerate(pdata["position"]):
                        positionData.append(
                            (position["x"], position["y"])
                        )
                    if len(positionData) <= 1:
                        continue
                    draw.polygon((positionData), fill='#ffffff')

            if len(brushData) > 0:
                for bdata in brushData:
                    positionData = []
                    draw = ImageDraw.Draw(mask)
                    lineWidth = bdata["lineWidth"]
                    position = bdata["points"]
                    mode = bdata["mode"]

                    for i in range(0, len(position), 2):
                        x = position[i]
                        y = position[i + 1]
                        positionData.append(
                            (x, y)
                        )

                    if mode == 'source-over':
                        draw.line((positionData), fill='#ffffff', joint='curve', width=lineWidth)
                    elif mode == 'destination-out':
                        draw.line((positionData), fill='#000000', joint='curve', width=lineWidth)

            mask.save(maskPath)

        makeMaskTotalTime = time.time() - makeMaskTime
        logger.info(f"Finish Making Mask of Test Data, Duration : {round(makeMaskTotalTime, 4)} sec")

    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())

        sys.exit(1)


def makeClassInfo(pretrained, originWeightPath, classInfo):

    '''
        정보 1. classes.json, classes.names 를 만들기 위해 리스트를 로직
    '''

    classIdList = []
    classNameList = []
    colorList = []
    preNumClasses = 0
    preChannel = 0

    try:
        if pretrained:
            originClassNameList = []
            with open(os.path.join(originWeightPath, "classes.json"), "r") as jsonFile:
                classesJson = json.load(jsonFile)

            for classesInfo in classesJson["classInfo"]:
                originClassNameList.append(classesInfo["className"])

            preNumClasses = len(originClassNameList)
            preChannel = int(classesJson["imageInfo"]["imageChannel"])

        for _class in classInfo:
            classIdList.append(_class["classId"].replace("'", ""))
            classNameList.append(_class["className"].replace("'", ""))
            colorList.append(_class["color"].replace("'", ""))
        
        classIdList = list(dict.fromkeys(classIdList))
        classNameList = list(dict.fromkeys(classNameList))
        colorList = list(dict.fromkeys(colorList))
        print("#############",classNameList)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())

        sys.exit(1)
    
    return classIdList, classNameList, colorList, preChannel, preNumClasses


# classes.names, classes.json 생성
def makeClasses(weightPath, classNameList, classIdList, colorList, imageSize, grayScale, purposeType):

    '''
        정보 1. classes.json, classes.names 를 만들기 위한 로직
    '''
    
    try:
         #open()구문을 사용하여 밑의 코드를 이용하면 classes.names파일을 쓰기용으로 만들 수 있다
        with open(os.path.join(weightPath, "classes.names"), "w") as f:
            f.writelines('\n'.join(classNameList))

        classesJsonFile = os.path.join(weightPath, "classes.json")

        classInfo = []

        for index in range(len(classNameList)):
            result = {
                "classId": classIdList[index],
                "className": classNameList[index],
                "color": colorList[index]
            }
            classInfo.append(result)

            saveJsonData = {
                "imageInfo": {
                    "imageSize": imageSize,
                    "imageChannel": grayScale,
                },
                "classInfo": classInfo,
                "purposeType": purposeType
            }
        #open()구문을 사용하여 밑의 코드를 이용하면 classes.json파일을 쓰기용으로 만들 수 있다.
        with open(classesJsonFile, "w") as f:
            json.dump(saveJsonData, f)
    
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        
        sys.exit(1)