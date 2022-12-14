import logging
import os
import pickle
import threading
from time import sleep

import cv2 as cv
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tqdm import tqdm

logger = logging.getLogger('ftpuploader')
logging.basicConfig(level=logging.INFO,
                    format='\n%(asctime)s :: %(levelname)s :: %(message)s')

RESIZED_IMAGE_WIDTH = 500
RESIZED_IMAGE_HEIGHT = 500
RESIZED_TUPLE = (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT)
RESIZED_PRODUCT = RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT
NUM_TREES = 200
SEED = 13
TEST_SIZE = 0.15


class Model:

    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=NUM_TREES, random_state=SEED)

        self.model_path = "model.sav"

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None

        self.cats_dir = "cats"
        self.cats_images = os.listdir(self.cats_dir)
        self.cats_img_list = np.array([])
        self.cats_progress = None

        self.dogs_dir = "dogs"
        self.dogs_images = os.listdir(self.dogs_dir)
        self.dogs_img_list = np.array([])
        self.dogs_progress = None

    def train_cats(self, cats_counter: int) -> None:
        logger.info("Starting the training with cat images...")
        self.cats_progress = tqdm(
            total=cats_counter, position=0, leave=False)
        for filename in self.cats_images:
            sleep(0.01)
            f = os.path.join(self.cats_dir, filename)
            if os.path.isfile(f):
                img = cv.imread(f)[:, :, 0]
                img_resized = cv.resize(img, RESIZED_TUPLE)
                img_reshaped = img_resized.reshape((1, RESIZED_PRODUCT))
                self.cats_img_list = np.append(self.cats_img_list, [img_reshaped])
                self.cats_progress.set_description(
                    "Training with cat images...".format(self.cats_images.index(filename)))
                self.cats_progress.update(1)
        self.cats_progress.close()
        logger.info("Done the training with cat images...")

    def train_dogs(self, dogs_counter: int) -> None:
        logger.info("Starting the training with dog images...")
        self.dogs_progress = tqdm(
            total=dogs_counter, position=0, leave=False)
        for filename in self.dogs_images:
            sleep(0.01)
            f = os.path.join(self.dogs_dir, filename)
            if os.path.isfile(f):
                img = cv.imread(f)[:, :, 0]
                img_resized = cv.resize(img, RESIZED_TUPLE)
                img_reshaped = img_resized.reshape((1, RESIZED_PRODUCT))
                self.dogs_img_list = np.append(self.dogs_img_list, [img_reshaped])
                self.dogs_progress.set_description(
                    "Training with dog images...".format(self.dogs_images.index(filename)))
                self.dogs_progress.update(1)
        self.dogs_progress.close()
        logger.info("Done the training with dog images...")

    def train_test_splitter(self, X, Y) -> None:
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y,
                                                                                test_size=TEST_SIZE,
                                                                                random_state=SEED)

    def fit_model(self) -> None:
        self.model.fit(self.X_train, self.Y_train)

    def train_model(self) -> None:
        if os.path.isfile(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
            logger.info(f"Model loaded from file {self.model_path}")
        else:
            cats_counter = len([file for file in os.listdir(self.cats_dir) if
                                os.path.isfile(os.path.join(self.cats_dir, file))])
            dogs_counter = len([file for file in os.listdir(self.dogs_dir) if
                                os.path.isfile(os.path.join(self.dogs_dir, file))])

            threads = list()
            threads.append(threading.Thread(target=self.train_cats, args=(cats_counter,), daemon=True))
            threads.append(threading.Thread(target=self.train_dogs, args=(dogs_counter,), daemon=True))
            for thread in threads: thread.start()
            for thread in threads: thread.join()

            logger.info("Finishing the training...")

            cats_class_list = np.repeat(1, cats_counter)
            dogs_class_list = np.repeat(2, dogs_counter)

            logger.info("Learning more about cats and dogs...")

            X = np.concatenate((self.cats_img_list, self.dogs_img_list), axis=None)
            Y = np.concatenate((cats_class_list, dogs_class_list), axis=None)

            X = X.reshape(cats_counter + dogs_counter, RESIZED_PRODUCT)

            split_thread = threading.Thread(target=self.train_test_splitter, args=(X, Y,), daemon=True)
            split_thread.start()
            split_thread.join()

            logger.info("Doing the IA magic stuff...")

            fit_thread = threading.Thread(target=self.fit_model(), daemon=True)
            fit_thread.start()
            fit_thread.join()

            logger.info("Saving the model...")

            pickle.dump(self.model, open(self.model_path, 'wb'))

            logger.info(
                f"Model successfully trained and saved as {self.model_path}!")
            logger.info(
                f"The accuracy of the model is {self.verify_accuracy()}")

    def predict(self, image: str) -> float:
        img = cv.imread(image)[:, :, 0]
        img_resized = cv.resize(img, RESIZED_TUPLE)
        img_reshaped = img_resized.reshape(RESIZED_PRODUCT)

        prediction = self.model.predict([img_reshaped])

        return prediction[0]

    def verify_accuracy(self) -> float:
        return self.model.score(self.X_test, self.Y_test)
