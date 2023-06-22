import os
import sys
import cv2
import json
import time
import torch
import traceback
from PIL import Image
from loguru import logger
from torchvision import transforms
from pytorch_grad_cam import GradCAMPlusPlus

from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
basePath = os.path.abspath(os.path.join(os.path.join(os.path.dirname(__file__), os.path.pardir)))
sys.path.extend([basePath, os.path.join(basePath, "model")])

from model import createModel
from logger import Logger
from checker import gpuChecker, initGpuChecker

gpuNo = initGpuChecker()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpuNo


class Predictor():
    def __init__(self, pathInfo):

        self.pathInfo = pathInfo
        self.modelPath = self.pathInfo["modelPath"] if "modelPath" in self.pathInfo else '/app'
        self.weightPath = self.pathInfo["weightPath"] if "weightPath" in self.pathInfo else "/app/weight"
        self.log = Logger(logPath=os.path.join(self.modelPath, "log/predict.log"), logLevel="info")

        # set cpu/gpu
        self.setGpu()

        if os.path.isfile(os.path.join(self.weightPath, "weight.pth")):
            with open(os.path.join(self.weightPath, "classes.json"), "r") as jsonFile:
                self.classesJson = json.load(jsonFile)

            self.classNameList = [classInfo["className"] for classInfo in self.classesJson["classInfo"]]
            self.imgSize = self.classesJson["imageInfo"]["imageSize"] if "imageSize" in self.classesJson["imageInfo"] else 224
            self.grayScale = int(self.classesJson["imageInfo"]["imageChannel"])

            if self.grayScale == 1:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imgSize, self.imgSize)),
                    transforms.Grayscale(num_output_channels=self.grayScale),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=0.5, std=0.5)
                ])

            else:
                self.transform = transforms.Compose([
                    transforms.Resize((self.imgSize, self.imgSize)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])

            # model load
            logger.info("Model Loading ...")

            modelLoadStartTime = time.time()

            self.model = createModel(
                pretrained=False,
                channel=self.grayScale,
                numClasses=len(self.classNameList)
            )
            self.model.eval()
            if torch.cuda.is_available():
                ckpt = torch.load(os.path.join(self.weightPath,"weight.pth"))
            else:
                ckpt = torch.load(os.path.join(self.weightPath,"weight.pth"), map_location=torch.device('cpu'))

            model_dict = self.model.state_dict()
            for key in model_dict.keys():
                if key in ckpt:
                    model_dict[key] = ckpt[key]

            self.model.load_state_dict(model_dict)
            self.model.eval()
            self.model.to(self.device)
            
            target_layer = [
                            self.model.patch_embed.proj,
                            self.model.layers[2].blocks[1].modulation.focal_layers[2],
                            self.model.layers[3].blocks[1].modulation.focal_layers[2],                            
                            ]

            self.gradCamModel = GradCAMPlusPlus(model=self.model, target_layers=target_layer, use_cuda=torch.cuda.is_available())

            modelLoadTime = time.time() - modelLoadStartTime
            logger.debug(f"Model Load Success, Duration : {round(modelLoadTime, 4)} sec")

        else:
            raise Exception("This Model is not Trained Model, Not Found Model's Weight File")

    def setGpu(self):
        self.device, self.deviceType = gpuChecker(log=self.log, gpuIdx="auto")

    def runPredict(self, predImage):

        try:
            logger.info("Starting Model Predict...")
            logger.info("-"*100)
            logger.info("  Device:             {}  ".format(self.device.type))
            logger.info("  Image Scaling:      {}  ".format((self.imgSize, self.imgSize, self.grayScale)))
            logger.info("  Labels:             {}  ".format(self.classNameList))

            result = []
            totalStartTime = time.time()

            # 이미지 예측을 위한 전처리
            logger.info("Input Data Preprocessing for Model...")
            preProStartTime = time.time()

            result = []
            originImage = predImage.copy()
            height, width = originImage.shape[:2]

            if self.grayScale == 1:
                predImage = cv2.cvtColor(predImage, cv2.COLOR_BGR2GRAY)
            else:
                predImage = cv2.cvtColor(predImage, cv2.COLOR_BGR2RGB)

            predImage = Image.fromarray(predImage)
            predImage = self.transform(predImage)
            predImage = predImage.unsqueeze(0)

            predImage = predImage.to(self.device)

            preProTime = time.time() - preProStartTime
            logger.debug(f"Input Data Preprocessing Success, Duration : {round(preProTime, 4)} sec")

            # 이미지 예측시작
            logger.info("Predict Start...")

            predStartTime = time.time()
            with torch.no_grad():
                predict = self.model(predImage)

            predTime = time.time() - predStartTime
            logger.debug(f"Predict Success, Duration : {round(predTime, 4)} sec")

            # 예측 결과 형태 변환
            transferOutputStartTime = time.time()
            logger.info("Output Format Transfer...")

            probabilities = torch.nn.functional.softmax(predict[0], dim=0)
            targetClasses = torch.argmax(probabilities)

            yPred = self.classNameList[targetClasses]
            accuracy = torch.max(probabilities)

            tmpResult = {
                "className": yPred,
                "cursor": "isText",
                "accuracy": float(accuracy.item()),
                "needCount": -1
            }

            targets = [ClassifierOutputTarget(int(targetClasses))]
            gradCamResult = self.gradCamModel(input_tensor=predImage, targets=targets)
            gradCamResult = gradCamResult[0, :]
            gradCamResult = cv2.resize(gradCamResult, (width, height), interpolation=cv2.INTER_CUBIC)
            gradCamResult = cv2.normalize(gradCamResult, gradCamResult, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatMapImage = cv2.applyColorMap(gradCamResult, cv2.COLORMAP_JET)

            logger.debug(tmpResult)
            result.append(tmpResult)

            trasferTime = time.time() - transferOutputStartTime
            logger.debug(f"Output Format Transfer Success, Duration : {round(trasferTime, 4)} sec")

            totalTime = time.time() - totalStartTime
            logger.info(f"Finish Model Predict, Duration : {round(totalTime, 4)} sec")
            logger.info("-"*100)

        except Exception as e:
            logger.error(f"Error :{str(e)}")
            logger.error(f"Traceback : {str(traceback.format_exc())}")

        return result, heatMapImage

if __name__ == "__main__":
    pathInfo = {
        "modelPath": "/data/sungmin/focal_test",
        "weightPath": "/data/sungmin/focalnet/save_weight",
        }

    img = cv2.imread("aaaa/aaaa/a.jpg")
    p = Predictor(pathInfo)

    # while True:
    predResult, heatMapImage = p.runPredict(img)
    print(predResult)
    cv2.imwrite("./heatmap.png", heatMapImage)