import cv2 as cv
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import logging
import pickle

logger = logging.getLogger('ftpuploader')
logging.basicConfig(level=logging.INFO, format='\n%(asctime)s :: %(levelname)s :: %(message)s')

RESIZED_IMAGE_WIDTH = 150
RESIZED_IMAGE_HEIGHT = 150


class Model:

    def __init__(self):
        self.model = LinearSVC(dual=False)

        self.model_path = "model.sav"

        self.X_train = None
        self.X_test = None
        self.Y_train = None
        self.Y_test = None
        self.test_size = 0.33
        self.seed = 13

        self.cats_train_dir = "training_set/cats"
        self.cats_test_dir = "test_set/cats"
        self.cats_train_images = os.listdir(self.cats_train_dir)
        self.cats_test_images = os.listdir(self.cats_test_dir)
        self.cats_progress = None

        self.dogs_train_dir = "training_set/dogs"
        self.dogs_test_dir = "test_set/dogs"
        self.dogs_train_images = os.listdir(self.dogs_train_dir)
        self.dogs_test_images = os.listdir(self.dogs_test_dir)
        self.dogs_progress = None

    def train_model(self):
        if os.path.isfile(self.model_path):
            self.model = pickle.load(open(self.model_path, 'rb'))
            logger.info(f"Model loaded from file {self.model_path}")
        else:
            cats_img_list = np.array([])
            dogs_img_list = np.array([])

            logger.info("Starting the training with cat images...")

            # train cats
            cats_train_counter = len([file for file in os.listdir(self.cats_train_dir) if
                                 os.path.isfile(os.path.join(self.cats_train_dir, file))])
            self.cats_progress = tqdm(total=cats_train_counter, position=0, leave=False)
            for filename in self.cats_train_images:
                f = os.path.join(self.cats_train_dir, filename)
                if os.path.isfile(f):
                    img = cv.imread(f)[:, :, 0]
                    img_resized = cv.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                    img_reshaped = img_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    cats_img_list = np.append(cats_img_list, [img_reshaped])
                    self.cats_progress.set_description(
                        "Training with cat images...".format(self.cats_train_images.index(filename)))
                    self.cats_progress.update(1)

            logger.info("Done the training with cat images...")
            logger.info("Starting the training with dog images...")

            # train dogs
            dogs_train_counter = len([file for file in os.listdir(self.dogs_train_dir) if
                                 os.path.isfile(os.path.join(self.dogs_train_dir, file))])
            self.dogs_progress = tqdm(total=dogs_train_counter, position=0, leave=False)
            for filename in self.dogs_train_images:
                f = os.path.join(self.dogs_train_dir, filename)
                if os.path.isfile(f):
                    img = cv.imread(f)[:, :, 0]
                    img_resized = cv.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
                    img_reshaped = img_resized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                    dogs_img_list = np.append(dogs_img_list, [img_reshaped])
                    self.dogs_progress.set_description(
                        "Training with dog images...".format(self.dogs_train_images.index(filename)))
                    self.dogs_progress.update(1)

            logger.info("Done the training with dog images...")

            self.cats_progress.close()
            self.dogs_progress.close()

            logger.info("Finishing the training...")

            cats_class_list = np.repeat(1, cats_train_counter)
            dogs_class_list = np.repeat(2, dogs_train_counter)

            logger.info("Doing the IA magic stuff...")

            X = np.concatenate((cats_img_list, dogs_img_list), axis=None)
            Y = np.concatenate((cats_class_list, dogs_class_list), axis=None)

            X = X.reshape(cats_train_counter + dogs_train_counter,
                                        RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)

            logger.info("Learning more about cats and dogs...")

            self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(X, Y,
                                                                                    test_size=self.test_size,
                                                                                    random_state=self.seed)

            logger.info("Fitting the model and saving it...")

            self.model.fit(self.X_train, self.Y_train)

            pickle.dump(self.model, open(self.model_path, 'wb'))

            logger.info(f"Model successfully trained and saved as {self.model_path}!")
            logger.info(f"The accuracy of the model is {self.verify_accuracy()}")

    def predict(self, image: str) -> float:
        img = cv.imread(image)[:, :, 0]
        img_resized = cv.resize(img, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))
        img_reshaped = img_resized.reshape(RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT)

        prediction = self.model.predict([img_reshaped])

        return prediction[0]

    def verify_accuracy(self) -> float:
        return self.model.score(self.X_test, self.Y_test)
