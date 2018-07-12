import random
import os
import sys
from keras import models
from keras.models import model_from_json
from keras import layers
from keras import losses
from keras import optimizers
import matplotlib.pyplot as plt
import numpy as np



maxV = 322
maxA = 6.5
trainDataset = []
validDataset = []
# [                                 bosch,  lampe,  ohp_halb,   ohp_voll,   laptop]
keyToLabel = {
    'bosch-single':                 [1,     0,      0,          0,          0],
    'lampe-ohp_voll':               [0,     1,      0,          1,          0],
    'lampe-single':                 [0,     1,      0,          0,          0],
    'laptop-single':                [0,     0,      0,          0,          1],
    'ohp_halb-laptop':              [0,     0,      1,          0,          1],
    'ohp_halb-laptop-lampe':        [0,     1,      1,          0,          1],
    'ohp_halb-single':              [0,     0,      1,          0,          0],
    'ohp_voll-laptop':              [0,     0,      0,          1,          1],
    'ohp_voll-laptop-bosch-lampe':  [1,     1,      0,          1,          1],
    'ohp_voll-single':              [0,     0,      1,          0,          0]
}
amount_epochs = 25


def save_model_to_JSON(model, json_name, weights_name):
    model_json = model.to_json()
    with open(json_name, 'w') as file:
        file.write(model_json)
    model.save_weights(weights_name)


def load_model_from_JSON(json_name, weights_name):
    model_json = open(json_name, 'r')
    loaded_model_json = model_json.read()
    model_json.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(weights_name)
    return loaded_model


def apply_model_on(model, pathToCSV):
    f = open(pathToCSV, 'r')
    f.readline()
    header = f.readline()
    f.readline()
    units = header.replace('\n', '').split(',')

    ampereFactor = 1
    if units[2] == '(mA)':
        ampereFactor = 1 / 1000

    lines = map(lambda l: l.replace('\n', ''), f.readlines())
    lines = map(lambda l: l.split(','), lines)
    lines = map(lambda l: list(map(lambda x: float(x), l)), lines)
    lines = list(map(lambda l: [l[0], l[1], l[2] * ampereFactor], lines))
    lines = map(lambda x: [x[1] / maxV, x[2] / maxA], lines)
    d = [float(y) for x in lines for y in x]
    test = np.zeros((1, len(d)))
    for col, val in enumerate(d):
        test[0, col] = val
    f.close()
    return model.predict(test)


def printRun(history_dict):
    print(history_dict)
    amount_epochs = len(history_dict['acc'])
    plt.plot(range(1, amount_epochs + 1), history_dict['loss'], 'ro', label='Training Set Loss')
    plt.plot(range(1, amount_epochs + 1), history_dict['val_loss'], 'r', label='Validation Set Loss')

    plt.xlabel('Training Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

    plt.plot(range(0, amount_epochs), history_dict['acc'], 'bo', label='Training Set Accuracy')
    plt.plot(range(0, amount_epochs), history_dict['val_acc'], 'b', label='Validation Set Accuracy')

    plt.xlabel('Training Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()


def create_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(4890,)))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(5, activation='sigmoid'))

    model.compile(optimizer=optimizers.Adam(),
                  loss=losses.binary_crossentropy,
                  metrics=['accuracy'])
    return model


def load_data(basedir):
    # Made by The BOSS
    data = {}

    vagina = os.listdir(basedir)  # top!
    for penis in vagina:
        print('load' + penis)
        curDir = os.path.join(basedir, penis)
        files = os.listdir(curDir)

        data[penis] = []
        for curFile in files:
            f = open(os.path.join(curDir, curFile), 'r')

            f.readline()
            header = f.readline()
            f.readline()

            units = header.replace('\n', '').split(',')

            # Mr. Unfähig ^2
            ampereFactor = 1
            if units[2] == '(mA)':
                ampereFactor = 1 / 1000

            lines = map(lambda l: l.replace('\n', ''), f.readlines())
            lines = map(lambda l: l.split(','), lines)
            lines = map(lambda l: list(map(lambda x: float(x), l)), lines)
            lines = list(map(lambda l: [l[0], l[1], l[2] * ampereFactor], lines))

            data[penis].append(lines)

            f.close()

    return data


def train_model(model, dir='./data'):
    random.seed(a=42)
    data = load_data(dir)
    for (key, val) in data.items():
        label = keyToLabel[key]
        for lines in val:
            lines = map(lambda x: [x[1] / maxV, x[2] / maxA], lines)
            d = [float(y) for x in lines for y in x]

            if random.random() < 0.8:
                trainDataset.append((label, d))
            else:
                validDataset.append((label, d))

    trainSetX = np.zeros((len(trainDataset), len(trainDataset[0][1])))
    trainSetY = np.zeros((len(trainDataset), 5))
    validSetX = np.zeros((len(validDataset), len(validDataset[0][1])))
    validSetY = np.zeros((len(validDataset), 5))

    for row, d in enumerate(trainDataset):
        for col, val in enumerate(d[0]):
            trainSetY[row, col] = val
        for col, val in enumerate(d[1]):
            trainSetX[row, col] = val

    for row, d in enumerate(validDataset):
        for col, val in enumerate(d[0]):
            validSetY[row, col] = val
        for col, val in enumerate(d[1]):
            validSetX[row, col] = val

    training = model.fit(x=trainSetX,
                         y=trainSetY,
                         epochs=amount_epochs,
                         batch_size=32,
                         shuffle=True,
                         validation_data=(validSetX, validSetY))
    printRun(training.history)
    res = model.evaluate(validSetX, validSetY)

    print("result on the testset: accuracy={}".format(res[1]))

# EPIC REAL MAIN WITHOUT GUI OF DOOM SHIT MASTER OF DESASTER


if __name__ == "__main__":
    if len(sys.argv) > 1:
        op = sys.argv[1].lower()
    else:
        op = 'help'

    if op == 'create':
        data = sys.argv[2]
        model_path = sys.argv[3]
        weights_path = sys.argv[4]

        model = create_model()
        train_model(model, data)
        save_model_to_JSON(model, model_path, weights_path)

    elif op == 'predict':
        model_path = sys.argv[2]
        weights_path = sys.argv[3]
        data_path = sys.argv[4]

        model = load_model_from_JSON(model_path, weights_path)

        print(apply_model_on(model, data_path))
        print('[bosch, lampe, ohp_halb, ohp_voll, laptop]')

    elif op == 'help':
        print()
        print("HILFE!!1 ich bin einer Glückskeksfabrik gefangen!!")
        print()
        print('-----------------------------------------------|>')
        print('train and save model:')
        print('python main.py create <data_dir> <model_path> <weights_path>')
        print()
        print('-----------------------------------------------|>')
        print('load model with weights to predict what kind of data is sampled in <data_path>')
        print('python main.py predict <model_path> <weights_path> <data_path>')
        print()
        print('-----------------------------------------------|>')
        print('prints this text and exits')
        print('python main.py help')
