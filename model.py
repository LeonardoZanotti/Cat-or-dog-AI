import cv2 as cv
import PIL
import numpy as np
from sklearn.svm import LinearSVC
import os


class Model:

    def __init__(self):
        self.model = LinearSVC()
        self.cats_test_dir = "test_set/cats"
        self.cats_train_dir = "training_set/cats"
        self.dogs_test_dir = "test_set/dogs"
        self.dogs_train_dir = "training_set/dogs"

    def train_model(self):
        img_list = np.array([])
        class_list = np.array([])

        # train cats
        for filename in os.listdir(self.cats_test_dir):
            f = os.path.join(self.cats_test_dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)

        # train dogs
        for filename in os.listdir(self.dogs_dir):
            f = os.path.join(self.dogs_dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                print(f)

        # for i in range(1, counters[0]):
        #     img = cv.imread(f"cats/{i}.jpg")[:,:,0]
        #     img = img.reshape(16950)
        #     img_list = np.append(img_list, [img])
        #     class_list = np.append(class_list, 1)
        #
        # for i in range(1, counters[1]):
        #     img = cv.imread(f"2/{i}.jpg")[:,:,0]
        #     img = img.reshape(16950)
        #     img_list = np.append(img_list, [img])
        #     class_list = np.append(class_list, 2)

        # img_list = img_list.reshape(counters[0] - 1 + counters[1] - 1, 16950)
        # self.model.fit(img_list, class_list)
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