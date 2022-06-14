import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from PIL import Image

X = np.load('image.npz')['arr_0']

y = pd.read_csv("labels.csv")["labels"]

print(pd.Series(y).value_counts())

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J' 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

nclasses = len(classes)

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state = 1, train_size = 7500, test_size = 2500)
x_trainScaled = x_train/255
x_testScaled = x_test/255

clf = LogisticRegression(solver = 'saga', multi_class = 'multinomial').fit(x_trainScaled, y_train)

def get_prediction(image):
    im_pil = Image.open(image)
    imagebw = im_pil.convert('L')
    
    imageResized = imagebw.resize((28, 28), Image.ANTIALIAS)
    pixelFilter = 20

    min_pixel = np.percentile(imageResized, pixelFilter)
    imageInverted = np.clip(imageResized-min_pixel, 0, 255)

    max_pixel = np.max(imageResized)
    imageInverted = np.asarray(imageInverted/max_pixel)

    test_sample = np.array(imageInverted).reshape(1, 784)
    test_predict = clf.predict(test_sample)
    return test_predict[0]