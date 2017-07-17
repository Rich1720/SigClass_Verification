import keras
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
from keras.optimizers import SGD, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import *
import numpy as np
import argparse
import cv2
import os
from imutils import paths

# Model constructor
class Architecture:
    @staticmethod
    def build(width, height, depth, classes, weightsPath):
        
        model = Sequential()
################################################
        model.add(Convolution2D(128, 5, 5, border_mode = "same", activation = "relu", input_shape=(height, width, depth)))
        model.add(Dropout(0.50, name = "Drop1"))
        
        model.add(Flatten(name = "Flat"))
        model.add(Dense(96, activation = 'relu', name = "Full1"))
        model.add(Dropout(0.25, name = "Drop3"))
        model.add(Dense(54, activation = 'relu', name = "Full2"))
        model.add(Dropout(0.25, name = "Drop4"))
        model.add(Dense(classes, activation = 'softmax'))
##############################################
        if weightsPath is not None:
            model.load_weights(weightsPath, by_name = True)
        return model

#Parsing arguments; save or loads weights from specified file, also how the path to dataset is entered
parser = argparse.ArgumentParser()
parser.add_argument("-s", "--save-model", type = int, default = -1, help="(optional) whether or not model should be saved to disk")
parser.add_argument("-l", "--load-model", type = int, default = -1, help="(optional) whether or not pre-trained model should be loaded")
parser.add_argument("-w", "--weights", type = str, help = "(optional) path to store weights")
parser.add_argument("-d", "--dataset", type = str, required = True, help = "path to the dataset")
args = vars(parser.parse_args())

#Establishing image paths
print("[STATUS] Dataset loading...\n\n")
imagePaths = list(paths.list_images(args["dataset"]))

data = []
counter = 0
labels = []
labelTypes = []
toMerge = cv2.imread("check_line.png")
alpha = 0.5
beta = 1 - alpha
#Basic filter to emphasize signatures from background
edgeDetect = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
#Goes through every image in path
for(i, imagePath) in enumerate(imagePaths):
    image = cv2.imread(imagePath)
    filtered = cv2.filter2D(src = image, ddepth = -1, kernel = edgeDetect)
    merged = addWeighted(image, alpha, toMerge, beta, 0.0)
    #Removes color
    grayScale = cv2.cvtColor(merged, cv2.COLOR_BGR2GRAY)
    label = imagePath.split(os.path.sep)[-1].split("_")[0]
    if label not in labelTypes:
        labelTypes.append(label)
    features = cv2.resize(grayScale, (150, 230)).flatten()
    data.append(features)
    labels.append(label)
    counter += 1
    if i > 0 and i % 100 == 0:
        print("PROCESSED {}/{}".format(i, len(imagePaths)))

LE = LabelEncoder()
labels = LE.fit_transform(labels)
#Reshape data to fir 2D Convolution layer
data = np.array(data).reshape(counter, 150, 230)
data = data[:, :, :, np.newaxis]
print(data.shape)


(trainData, testData, trainLabels, testLabels) = train_test_split(data / 255.0, labels, test_size = 0.25, random_state = 45)

trainLabels = np_utils.to_categorical(trainLabels, len(labelTypes))
testLabels = np_utils.to_categorical(testLabels, len(labelTypes))

#Passing input dimensions
print("[STATUS] Model is being compiled...")
model = Architecture.build(width = data[0][0].size, height = data[0].size / data[0][0].size, depth = data[0][0][0].size, classes = len(labelTypes), weightsPath = args["weights"] if args["load_model"] > 0 else None)
print(model.summary())
model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.01), metrics = ["accuracy"])
model.fit(x = trainData, y = trainLabels, batch_size=32, epochs=20, verbose=1)
model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.001), metrics = ["accuracy"])
model.fit(x = trainData, y = trainLabels, batch_size=32, epochs=20, verbose=1)
model.compile(loss = "categorical_crossentropy", optimizer = SGD(lr = 0.0001), metrics = ["accuracy"])
model.fit(x = trainData, y = trainLabels, batch_size=32, epochs=20, verbose=1)

if args["load_model"] < 0:
    print("TRAINING WEIGHTS...")
    model.summary()
    model.fit(x = trainData, y = trainLabels, batch_size=64, epochs=30, verbose=1)
    print("EVALUATING DATA...")
    (loss, accuracy) = model.evaluate(testData, testLabels, batch_size=64, verbose=1)
    print("ACCURACY >> {:.4f}%".format(accuracy * 100))

if args["save_model"] > 0:
    print("Weights being stored for future use...")
    model.save_weights(args["weights"], overwrite=True)

for i in np.random.choice(np.arange(0, len(testLabels)), size = (len(labelTypes))):
    probs = model.predict(testData[np.newaxis, i])
    prediction = probs.argmax(axis = 1)
    print("PREDICTED SIGNATURE: {}, ACTUAL: {}".format(prediction[0], np.argmax(testLabels[i])))
