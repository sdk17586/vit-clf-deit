import os
import json
import time
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from loguru import logger
from torch.optim import Adam, AdamW, SGD
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import calculateAccuracy
from sender import sendMsg
from model import createModel


class CustomEarlyStopping(EarlyStopping):
    def on_validation_end(self, trainer, pl_module):
        # override this to disable early stopping at the end of val loop
        pass

    def on_train_end(self, trainer, pl_module):
        # instead, do it at the end of training loop
        self._run_early_stopping_check(trainer)


class Callback(pl.LightningModule):
    def __init__(self, param):
        super(Callback, self).__init__()

        self.param = param

        self.param.Log.info("Create Model...")
        createModelStartTime = time.time()

        self.model = createModel(
            pretrained=self.param.pretrained,
            preChannel=self.param.preChannel,
            channel=self.param.grayScale,
            preNumClasses=self.param.preNumClasses,
            numClasses=len(self.param.classNameList),
            weightPath=self.param.originWeightPath,
            device=self.param.device
        )

        createModelTime = time.time() - createModelStartTime
        self.param.Log.info(f"Finsh Create Model, Duration : {round(createModelTime, 4)} sec")

        self.model.names = list(self.param.classNameList)

        self.model.nc = len(self.param.classNameList)

        self.curEpoch = 1
        self.trainBatchEpoch = math.ceil(int(len(self.param.trainDataLoader.dataset)) / int(self.param.batchSize))
        self.validBatchEpoch = math.ceil(int(len(self.param.validDataLoader.dataset)) / int(self.param.batchSize))

        self.trainLoss = 0
        self.trainAccuracy = 0
        self.validLoss = 0
        self.validAccuracy = 0

        self.criterion = nn.CrossEntropyLoss()

        self.servInfo = self.param.servInfo
        self.sendStatusUrl = "{}:{}/{}".format(
            self.servInfo["servIp"],
            self.servInfo["servPort"],
            self.servInfo["sendStatusUrl"]
        )
    
        self.sendResultUrl = "{}:{}/{}".format(
            self.servInfo["servIp"],
            self.servInfo["servPort"],
            self.servInfo["sendResultUrl"]
        )

    def forward(self, image):
        predImage = self.model(image)

        return predImage

    def configure_callbacks(self):
        if self.param.earlyStopping:
            self.earlyStop = CustomEarlyStopping(monitor=self.param.monitor, mode="min", patience=self.param.patience, verbose=True)

            return [self.earlyStop]
        else:
            return None

    def configure_optimizers(self):
        lr = float(self.param.learningRate)

        if self.param.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=lr)
        elif self.param.optimizer == 'adamw':
            self.optimizer = AdamW(self.model.parameters(), lr=lr)
        elif self.param.optimizer == 'sgd':
            self.optimizer = SGD(params=self.model.parameters(), lr=lr)

    def train_dataloader(self):

        return self.param.trainDataLoader
    
    def val_dataloader(self):

        return self.param.validDataLoader

    def on_train_start(self):

        self.param.Log.info("Starting Training...")
        self.param.Log.info("   Epochs:             {}".format(self.param.epoch))
        self.param.Log.info("   Batch size:         {}".format(self.param.batchSize))
        self.param.Log.info("   Device:             {}".format(self.param.device.type))
        self.param.Log.info("   Optimizer:          {}".format(self.param.optimizer))
        self.param.Log.info("   LearningRate:       {}".format(self.param.learningRate))
        self.param.Log.info("   Image Scaling:      {}".format((self.param.imageSize, self.param.imageSize, self.param.grayScale)))
        self.param.Log.info("   Labels:             {}".format(self.param.classNameList))
        self.param.Log.info("   EarlyStopping:      {}".format(self.param.earlyStopping))
        self.param.Log.info("   Mode/Patience:      {} / {}".format(self.param.monitor, self.param.patience))

        self.param.Log.info("Training epoch={}/{}".format(
            self.curEpoch,
            self.param.epoch
        ))

        self.epochStartTime = time.time()

        logger.debug("Starting Training...")

        logger.debug("Training epoch={}/{}".format(
            self.curEpoch,
            self.param.epoch
        ))

        self.model.to(self.param.device)

    def training_step(self, batch, batch_idx):
        image, label = batch

        image = image.to(self.param.device)
        label = label.to(self.param.device)

        self.model.eval()
        yPred = self(image)
        self.model.train()
        trainBatchLoss = self.criterion(yPred, label)
        trainBatchAccuracy = calculateAccuracy(yPred, label)

        self.optimizer.zero_grad()
        trainBatchLoss.requires_grad_(True)
        trainBatchLoss.backward()
        self.optimizer.step()

        self.trainLoss += float(trainBatchLoss.item())
        self.trainAccuracy += float(trainBatchAccuracy.item())

        self.param.Log.info("Training batch={}/{}:{}, loss={}, accuracy={}".format(
            int(batch_idx + 1),
            int(self.trainBatchEpoch),
            int(self.curEpoch),
            float("{0:.4f}".format(float(trainBatchLoss.item()))),
            float("{0:.4f}".format(float(trainBatchAccuracy.item())))
        ))

        logger.debug("Training batch={}/{}:{}, loss={}, accuracy={}".format(
            int(batch_idx + 1),
            int(self.trainBatchEpoch),
            int(self.curEpoch),
            float("{0:.4f}".format(float(trainBatchLoss.item()))),
            float("{0:.4f}".format(float(trainBatchAccuracy.item())))
        ))

    def validation_step(self, batch, batch_idx):
        image, label = batch

        image = image.to(self.param.device)
        label = label.to(self.param.device)
        self.model.eval()


        with torch.no_grad():
            yPred = self(image)
            validLoss = self.criterion(yPred, label)
            validAccuracy = calculateAccuracy(yPred, label)

        self.validLoss += float(validLoss.item())
        self.validAccuracy += float(validAccuracy.item())

    def validation_epoch_end(self, outs):

        self.validLoss /= self.validBatchEpoch
        self.validAccuracy /= self.validBatchEpoch

        self.validLoss = round(float(self.validLoss), 4)
        self.validAccuracy = round(float(self.validAccuracy), 4) if float(self.validAccuracy) <= 1 else 1.0

        self.log("valLoss", self.validLoss)
        self.log("valAccuracy", self.validAccuracy)

    def training_epoch_end(self, outputs):

        self.trainLoss /= self.trainBatchEpoch
        self.trainAccuracy /= self.trainBatchEpoch

        self.trainLoss = round(float(self.trainLoss), 4)
        self.trainAccuracy = round(float(self.trainAccuracy), 4) if float(self.trainAccuracy) <= 1 else 1.0

        self.epochEndTime = time.time() - self.epochStartTime
        self.remaningTime = self.epochEndTime * (self.param.epoch - self.curEpoch)
        self.curEpoch += 1

        self.log("loss", self.trainLoss)
        self.log("accuracy", self.trainAccuracy)

    def on_train_epoch_end(self):
        self.param.Log.info("Result epoch={}/{}, loss={}, accuracy={}, valLoss={}, valAccuracy={}".format(
            self.curEpoch,
            self.param.epoch,
            self.trainLoss,
            self.trainAccuracy,
            self.validLoss,
            self.validAccuracy
        ))

        logger.debug("Result epoch={}/{}, loss={}, accuracy={}, valLoss={}, valAccuracy={}".format(
            self.curEpoch - 1,
            self.param.epoch,
            self.trainLoss,
            self.trainAccuracy,
            self.validLoss,
            self.validAccuracy
        ))

        trainTrialResult = {
            "epoch": self.curEpoch - 1,
            "purposeType": self.param.dataInfo["purposeType"],
            "mlType": self.param.dataInfo["mlType"],
            "loss": self.trainLoss,
            "valLoss": self.validLoss,
            "accuracy": self.trainAccuracy,
            "valAccuracy": self.validAccuracy,
            "isLast": False,
            "remaningTime": self.remaningTime,
            "elapsedTime": self.epochEndTime
        }

        if self.param.earlyStopping:
            self.param.Log.info(f"Early Stopping Patience={self.earlyStop.wait_count}/{int(self.param.patience)}")
            logger.debug(f"Early Stopping Patience={self.earlyStop.wait_count}/{int(self.param.patience)}")
            logger.debug(trainTrialResult)
            torch.save(self.model.state_dict(), self.param.modelWeightPath)

            if (self.earlyStop.wait_count == int(self.param.patience)) or (int(self.curEpoch - 1) == self.param.epoch):
                self.param.Log.info("Model is Stopped!")
                logger.debug("Model is Stopped!")

                trainTrialResult["isLast"] = True
                
                # Trial Send (isLast=True)
                # _ = sendMsg(self.sendStatusUrl, trainTrialResult)
 
                logger.debug("Model Weight Save Success!")

                trainResult = {
                    "score": round(float(self.validAccuracy), 4)  * 100,
                    "trainInfo": trainTrialResult
                }

                # FinalResult Send
                # _ = sendMsg(self.sendResultUrl, trainResult)

                tranResultPath = os.path.join(self.param.pathInfo["modelPath"], "trainResult.json")

                with open(tranResultPath, "w") as f:
                    json.dump(trainResult, f)
            
            else:
                # Trial Send (isLast=False)
                # _ = sendMsg(self.sendStatusUrl, trainTrialResult)
                print("ok")
        else:
            logger.debug(trainTrialResult)
            torch.save(self.model.state_dict(), self.param.modelWeightPath)
            
            if int(self.curEpoch - 1) == self.param.epoch:

                # Trial Send (isLast=True)
                trainTrialResult["isLast"] = True
                logger.debug("Model Weight Save Success!")

                # _ = sendMsg(self.sendStatusUrl, trainTrialResult)

                trainResult = {
                    "score": round(float(self.validAccuracy), 4) * 100,
                    "trainInfo": trainTrialResult
                }

                # FinalResult Send
                # _ = sendMsg(self.sendResultUrl, trainResult)

                tranResultPath = os.path.join(self.param.pathInfo["modelPath"], "trainResult.json")

                with open(tranResultPath, "w") as f:
                    json.dump(trainResult, f)

            else:
                # Trial Send (isLast=False)
                # _ = sendMsg(self.sendStatusUrl, trainTrialResult)
                print('ok')
        # 초기화
        self.trainLoss = 0
        self.trainAccuracy = 0
        self.validLoss = 0
        self.validAccuracy = 0