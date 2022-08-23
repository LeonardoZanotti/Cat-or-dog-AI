import cv2 as cv
import PIL, PIL.Image
import numpy as np
from sklearn.svm import LinearSVC
import os
from tqdm import tqdm
import logging

logger = logging.getLogger('ftpuploader')
logging.basicConfig(level=logging.INFO, format='\n%(asctime)s :: %(levelname)s :: %(message)s')

RESIZED_IMAGE_WIDTH = 150
RESIZED_IMAGE_HEIGHT = 150


class Model:

    def __init__(self):
        self.model = LinearSVC()

        self.cats_test_dir = "test_set/cats"
        self.cats_train_dir = "training_set/cats_min"
        self.cats_counter = len([file for file in os.listdir(self.cats_train_dir) if
                                 os.path.isfile(os.path.join(self.cats_train_dir, file))])
        self.cats_progress = None

        self.dogs_test_dir = "test_set/dogs"
        self.dogs_train_dir = "training_set/dogs_min"
        self.dogs_counter = len([file for file in os.listdir(self.dogs_train_dir) if
                                 os.path.isfile(os.path.join(self.dogs_train_dir, file))])
        self.dogs_progress = None

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        cats_directory = os.listdir(self.cats_train_dir)
        dogs_directory = os.listdir(self.dogs_train_dir)

        logger.info("Starting the training with cat images...")

        # train cats
        self.cats_progress = tqdm(total=self.cats_counter, position=0, leave=False)
        for filename in cats_directory:
            f = os.path.join(self.cats_train_dir, filename)
            if os.path.isfile(f):
                img = cv.imread(f)[:, :, 0]
                img_resized = cv.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                img_reshaped = img_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                img_list = np.append(img_list, [img_reshaped])
                class_list = np.append(class_list, 1)
                self.cats_progress.set_description("Training with cat images...".format(cats_directory.index(filename)))
                self.cats_progress.update(1)

        logger.info("Done the training with cat images...")
        logger.info("Starting the training with dog images...")

        # train dogs
        self.dogs_progress = tqdm(total=self.dogs_counter, position=0, leave=False)
        for filename in dogs_directory:
            f = os.path.join(self.dogs_train_dir, filename)
            if os.path.isfile(f):
                img = cv.imread(f)[:, :, 0]
                img_resized = cv.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                img_reshaped = img_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                img_list = np.append(img_list, [img_reshaped])
                class_list = np.append(class_list, 2)
                self.dogs_progress.set_description("Training with dog images...".format(dogs_directory.index(filename)))
                self.dogs_progress.update(1)

        logger.info("Done the training with cat images...")

        self.cats_progress.close()
        self.dogs_progress.close()

        img_list = img_list.reshape(self.cats_counter + self.dogs_counter, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)
        self.model.fit(img_list, class_list)
        logger.info("Model successfully trained!")

    def predict(self, image: str) -> float:
        img = cv.imread(image)[:, :, 0]
        img_resized = cv.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        img_reshaped = img_resized.reshape(RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)

        prediction = self.model.predict([img_reshaped])

        # logger.info(prediction[0])

        return prediction[0]
