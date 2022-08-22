import cv2 as cv
import PIL
import numpy as np
from sklearn.svm import LinearSVC
import os
from tqdm import tqdm


class Model:

    def __init__(self):
        self.model = LinearSVC()

        self.cats_test_dir = "test_set/cats"
        self.cats_train_dir = "training_set/1"
        self.cats_counter = len([file for file in os.listdir(self.cats_train_dir) if os.path.isfile(os.path.join(self.cats_train_dir, file))])
        self.cats_progress = tqdm(total=self.cats_counter, position=0, leave=False)

        self.dogs_test_dir = "test_set/dogs"
        self.dogs_train_dir = "training_set/2"
        self.dogs_counter = len([file for file in os.listdir(self.dogs_train_dir) if os.path.isfile(os.path.join(self.dogs_train_dir, file))])
        self.dogs_progress = tqdm(total=self.dogs_counter, position=0, leave=False)

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        cats_directory = os.listdir(self.cats_train_dir)
        dogs_directory = os.listdir(self.dogs_train_dir)

        # train cats
        for filename in cats_directory:
            f = os.path.join(self.cats_train_dir, filename)
            if os.path.isfile(f):
                img = cv.imread(f)[:, :, 0]
                # img = img.reshape(16950)
                img_list = np.append(img_list, [img])
                class_list = np.append(class_list, 1)
                self.cats_progress.set_description("Training with cat images...".format(cats_directory.index(filename)))
                self.cats_progress.update(1)

        # train dogs
        for filename in dogs_directory:
            f = os.path.join(self.dogs_train_dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                img = cv.imread(f)[:, :, 0]
                # img = img.reshape(16950)
                img_list = np.append(img_list, [img])
                class_list = np.append(class_list, 2)
                self.dogs_progress.set_description("Training with dog images...".format(dogs_directory.index(filename)))
                self.dogs_progress.update(1)

        self.cats_progress.close()
        self.dogs_progress.close()

        img_list = img_list.reshape(self.cats_counter + self.dogs_counter, 16950)
        self.model.fit(img_list, class_list)
        print("Model successfully trained!")

    def predict(self, frame):
        frame = frame[1]
        cv.imwrite("frame.jpg", cv.cvtColor(frame, cv.COLOR_RGB2GRAY))
        img = PIL.Image.open("frame.jpg")
        img.thumbnail((150, 150), PIL.Image.ANTIALIAS)
        img.save("frame.jpg")

        img = cv.imread("frame.jpg")[:,:,0]
        img = img.reshape(16950)
        prediction = self.model.predict([img])
        return prediction[0]