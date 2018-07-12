import random
import os
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as plt

#Made by The BOSS

basedir = './data'

data = {}

configurations = os.listdir(basedir)
for penis in configurations:
    print('load ' + penis)
    curDir = os.path.join(basedir, penis)
    files = os.listdir(curDir)

    data[penis] = []
    for curFile in files:
        #print('read' + curFile)
        f = open(os.path.join(curDir, curFile), 'r')

        f.readline()
        header = f.readline()
        f.readline()

        units = header.replace('\n', '').split(',')

        #Mr. Unfähig ^2
        ampereFactor = 1
        if units[2] == '(mA)':
            ampereFactor = 1/1000

        lines = map(lambda l: l.replace('\n', ''), f.readlines())
        lines = map(lambda l: l.split(','), lines)
        d = list(map(lambda l: [float(l[0]), float(l[1]), float(l[2])*ampereFactor], lines))
        data[penis].append(d)

        f.close()

import numpy as np

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
amount_epochs = 6
random.seed(a=42)

maxV = 0
maxA = 0
for (key, val) in data.items():
    for lines in val:
        for d in lines:
            if abs(d[1]) > maxV:
                maxV = abs(d[1])

            if abs(d[2]) > maxA:
                maxA = abs(d[2])

for (key, val) in data.items():
    label = keyToLabel[key]
    for d in val:
        d = list(map(lambda x: [x[1] / maxV, x[2] / maxA], d))
        if random.random() < 0.8:
            trainDataset.append((label, d))
        else:
            validDataset.append((label, d))

trainSetX = np.zeros((len(trainDataset), len(trainDataset[0][1]), 2))
trainSetY = np.zeros((len(trainDataset), 5))
validSetX = np.zeros((len(validDataset), len(validDataset[0][1]), 2))
validSetY = np.zeros((len(validDataset), 5))

for row, d in enumerate(trainDataset):
    for col, val in enumerate(d[0]):
        trainSetY[row, col] = val
    for col, arr in enumerate(d[1]):
        for z, val in enumerate(arr):
            trainSetX[row, col, z] = val

for row, d in enumerate(validDataset):
    for col, val in enumerate(d[0]):
        validSetY[row, col] = val
    for col, arr in enumerate(d[1]):
        for z, val in enumerate(arr):
            validSetX[row, col, z] = val

model = models.Sequential()
model.add(layers.LSTM(128, return_sequences=True))
model.add(layers.Dropout(0.2))
model.add(layers.LSTM(128))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(5, activation='sigmoid'))

model.compile(optimizer=optimizers.Adam(),
              loss=losses.binary_crossentropy,
              metrics=['accuracy'])

training = model.fit(x=trainSetX,
                     y=trainSetY,
                     epochs=amount_epochs,
                     batch_size=32,
                     shuffle=True,
                     validation_data=(validSetX, validSetY))

res = model.evaluate(validSetX, validSetY)

print("result on the testset: accuracy={}".format(res[1]))

history_dict = training.history
print(history_dict)
amount_epochs = len(history_dict['acc'])
plt.plot(range(1, amount_epochs+1), history_dict['loss'], 'ro', label='Training Set Loss')
plt.plot(range(1, amount_epochs+1), history_dict['val_loss'], 'r', label='Validation Set Loss')

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

# print(model.predict(textX))
