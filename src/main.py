import random
import os
from keras import models
from keras import layers
from keras import losses
from keras import optimizers

#Made by The BOSS

basedir = '../data'

data = {}

configurations = os.listdir(basedir)
for penis in configurations:
    print('load' + penis)
    curDir = os.path.join(basedir, penis)
    files = os.listdir(curDir)

    data[penis] = []
    for curFile in files:
        #print('read' + curFile)
        f = open(os.path.join(curDir, curFile), 'r')

        for x in [1, 2, 3]:
            f.readline()
        lines = map(lambda l: l.replace('\n', ''), f.readlines())
        lines = map(lambda l: l.split(','), lines)
        d = [y for x in lines for y in x]
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

for (key, val) in data.items():
    label = keyToLabel[key]
    for d in val:
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
        trainSetY[row, col] = val
    for col, val in enumerate(d[1]):
        validSetX[row, col] = val

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(len(trainSetX[0]),)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.RMSprop(), loss=losses.binary_crossentropy, metrics=['accuracy'])

model.fit(x=trainSetX, y=trainSetY, epochs=10, verbose=2, steps_per_epoch=10)

res = model.evaluate(validSetX, validSetY)

print("result on the testset: accuracy={}".format(res[1]))
