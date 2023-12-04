import os
import time
from functools import wraps

import logging
logging.basicConfig(level=logging.INFO)
logger=logging.getLogger(__name__)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from tensorflow.keras.models import load_model


from motionvector.Motion import diffCalForTestData

from signature.Model import preprocess_data


total_time=0

def time_it(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        global total_time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        total_time += elapsed_time
        print(f"Time taken by {func.__name__}: {elapsed_time} seconds")
        return result
    return wrapper
class Pipeline():
    def __init__(self, testFilePath):
        logger.info("Initializing Pipeline")
        self.testFilePath=testFilePath
        self.class_indices= None
        self.labels=None
        self.model=None
        self.testFrames =list()
        self.testMotionResidue=None
        self.newWidth=100
        self.newHeight=100


    @time_it
    def loadLabels(self,trainDir,testDir):
        logger.info("Loading Labels")
        train_generator, test_generator = preprocess_data(
            trainDir,
            testDir)

        self.class_indices = train_generator.class_indices
        self.labels = {v: k for k, v in self.class_indices.items()}

    @time_it
    def loadPreTrainedModel(self,savedModelPath):
        logger.info("Loading Pretrained Model")
        self.model = load_model(savedModelPath)

    @time_it
    def predict(self,frame):
        logger.info("Predicting video class")
        img = cv2.resize(frame, (100, 100))
        img = np.reshape(img, (1, 100, 100, 1))
        img = img / 255.0
        # Make a prediction
        prediction = self.model.predict(img)

        predicted_class = np.argmax(prediction, axis=1)
        logger.info("Predicted class: "+str(predicted_class[0]))
        return self.labels[predicted_class[0]]

    def downscale_image(self,original_frame, new_width, new_height):
        img_ycrcb = cv2.cvtColor(original_frame, cv2.COLOR_RGB2YCrCb)
        Y, Cr, Cb = cv2.split(img_ycrcb)
        downscaled_image = cv2.resize(Y, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return downscaled_image

    @time_it
    def extract_frames(self):
        logger.info("Extracting frames from video")
        vidcap = cv2.VideoCapture(self.testFilePath)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        success, image = vidcap.read()
        count = 0

        while success:
            # Save frame every second
            if count % int(fps) == 0:
                image = self.downscale_image(image, self.newWidth, self.newHeight)
                self.testFrames.append(image)
            success, image = vidcap.read()
            count += 1

        vidcap.release()
        logger.info("Done extracting Frames.")

    @time_it
    def extractMotionResidue(self):
        logger.info("Extracting Motion Residue")
        self.testMotionResidue=diffCalForTestData(self.testFilePath)
        logger.info("Done extracting Motion Residue")



if __name__ == "__main__":

    pipeline=Pipeline("/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Queries/video1_1.mp4")
    pipeline.loadLabels('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Train',
                        '/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/Test')
    pipeline.loadPreTrainedModel('/Users/sms/USC/MS-SEM2/multimedia/video-shazam/dataset/Data/SavedModel/signatureDetect.h5')
    pipeline.extract_frames()
    pipeline.extractMotionResidue()
    print(pipeline.predict(pipeline.testFrames[0]))
    logger.info("Total time taken: "+str(total_time))






