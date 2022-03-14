from cProfile import label
from re import X
from tabnanny import verbose
import numpy as np
from sklearn.utils import shuffle
from tensorflow import keras
from tensorflow.keras import layers

import struct
from PIL import Image
import numpy as np

import skimage.transform
import numpy as np
from sklearn.model_selection import train_test_split


from keras.preprocessing.image import ImageDataGenerator
from sklearn import preprocessing

## Model / data parameters
num_classes = 48
input_shape = (48, 48, 1)

## the data, split between train and test sets
#(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

## Scale images to the [0, 1] range
#x_train = x_train.astype("float32") / 255
#x_test = x_test.astype("float32") / 255
## Make sure images have shape (28, 28, 1)
#x_train = np.expand_dims(x_train, -1)
#x_test = np.expand_dims(x_test, -1)
#print("x_train shape:", x_train.shape)
#print(x_train.shape[0], "train samples")
#print(x_test.shape[0], "test samples")


## convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)





def read_record_ETL1G(f):
    s = f.read(2052)
    r = struct.unpack('>H2sH6BI4H4B4x2016s4x', s)
    iF = Image.frombytes('F', (64, 63), r[18], 'bit', 4)
    iL = iF.convert('P')
    return r + (iL,)

def read_kana():
    katakana = np.zeros([51, 1411, 63, 64], dtype=np.uint8) # 51 characters, 1411 writers, img size = 63*64
    for i in range(7,14):
        filename = 'ETL1/ETL1C_{:02d}'.format(i)
        with open(filename, 'rb') as f: # file 13 only has 3 characters, others have 8 characters
            if i!=13: limit = 8
            else: limit=3
            for dataset in range(limit):
                for j in range(1411):
                    try :
                        r = read_record_ETL1G(f)
                        katakana[(i - 7) * 8 + dataset, j] = np.array(r[-1])
                    except struct.error: # two imgs are blank according to the ETL website, so this prevents any errors
                        pass
    np.savez_compressed("kana.npz", katakana)

#read_kana()






kana = np.load("kana.npz")['arr_0'].reshape([-1, 63, 64]).astype(np.float32)

kana = kana/np.max(kana) # make the numbers range from 0 to 1

# 51 is the number of different katakana (3 are duplicates so in the end there are 48 classes), 1411 writers.
train_images = np.zeros([51 * 1411, 48, 48], dtype=np.float32)

for i in range(51 * 1411): # change the image size to 48*48
    train_images[i] = skimage.transform.resize(kana[i], (48, 48))

arr = np.arange(51) # create labels
train_labels = np.repeat(arr, 1411)

# In the actual code, I combined the duplicate classes here and had 48 classes in the end
for i in range(len(train_labels)):
	if train_labels[i] == 36:
		train_labels[i] = 1
	elif train_labels[i] == 38:
		train_labels[i] = 3
	elif train_labels[i] == 47:
		train_labels[i] = 2
	elif train_labels[i] == 37:
		train_labels[i] = train_labels[i] -1
	elif train_labels[i] >= 39 and train_labels[i] <= 46:
		train_labels[i] = train_labels[i] - 2
	elif train_labels[i] >= 48:
		train_labels[i] = train_labels[i] -3

delete = [] # the 33863th and 67727th images are blank, so we delete them
for i in range(len(train_images)):
	if (train_images[i] == np.zeros([train_images[i].shape[0],train_images[i].shape[1]],dtype=np.uint8) ).all():
		delete.append(i)

train_images = np.delete(train_images,delete[0],axis=0)
train_labels = np.delete(train_labels,delete[0])

train_images = np.delete(train_images,delete[1]-1,axis=0)
train_labels = np.delete(train_labels,delete[1]-1)

# split the images/labels to train and test
train_images, test_images, train_labels, test_labels = train_test_split(train_images, train_labels, test_size=0.2)

np.savez_compressed("katakana_train_images.npz", train_images)
np.savez_compressed("katakana_train_labels.npz", train_labels)
np.savez_compressed("katakana_test_images.npz", test_images)
np.savez_compressed("katakana_test_labels.npz", test_labels)


import matplotlib.pyplot as plt
plt.figure(figsize=(6,6)).patch.set_facecolor('#E56B51')
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
plt.show()

from keras import backend as K
if K.image_data_format() == "channels_first": # reshape the image to be able to go through 2D CNN
  train_images = train_images.reshape(train_images.shape[0], 1,48,48)
  test_images = test_images.reshape(test_images.shape[0], 1,48,48)
  input_shape = (1,48,48)
else:
  train_images = train_images.reshape(train_images.shape[0], 48, 48, 1)
  test_images = test_images.reshape(test_images.shape[0], 48, 48, 1)
  input_shape = (48,48,1)




# 4 conv, 1 dropout, softmax a la fin
datagen = ImageDataGenerator(rotation_range=15,zoom_range=0.2)
datagen.fit(train_images)
model = keras.Sequential(
    [
        keras.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ]
)

model.summary()


batch_size = 500
epochs = 25

#model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

#model.fit(datagen.flow(train_images, train_labels, shuffle = True), batch_size=batch_size, epochs=epochs, validation_data=(test_images,test_labels))

#model.save("katakana-model.h5")
model = keras.models.load_model("katakana-model.h5")
#model.summary()

score = model.evaluate(test_images, test_labels, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

img = "testDetectionE.png"
img = Image.open(img)
#plt.imshow(img)
#plt.show()
plt.gray()
img_bw = img.convert('L')
#plt.imshow(img_bw)
#plt.show()


img_array = np.array(img_bw)
img_batch = np.expand_dims(img_array, axis=0)
score2 = model.predict(img_batch)
print(score2)
print(score2.argmax(axis=-1)[0])

indexToLabel = {
    0 : "A - ア", 1 : "I - イ", 2 : "U - ウ", 3 : "E - エ", 4 : "O - オ",
    5 : "KA - カ", 6 : "KI - キ", 7 : "KU - ク", 8 : "KE - ケ", 9 : "KO - コ",
    10 : "SA - サ", 11 : "SHI - シ", 12 : "SU - ス", 13 : "SE - セ", 14 : "SO - ソ",
    15 : "TA - タ", 16 : "CHI - チ", 17 : "TSU - ツ", 18 : "TE - テ", 19 : "TO - ト",
    20 : "NA - ナ", 21 : "NI - ニ", 22 : "NU - ヌ", 23 : "NE - ネ", 24 : "NO - ノ",
    25 : "HA - ハ", 26 : "HI - ヒ", 27 : "FU - フ", 28 : "HE - ヘ", 29 : "HO - ホ",
    30 : "MA - マ", 31 : "MI - ミ", 32 : "MU - ム", 33 : "ME - メ", 34 : "MO - モ",
    35 : "YA - ヤ", 36 : "YU - ユ", 37 : "YO - ヨ", 38 : "RA - ラ", 39 : "RI - リ",
    40 : "RU - ル", 41 : "RE - レ", 42 : "RO - ロ", 43 : "WA - ワ", 44 : "WI - ヰ",
    45 : "WE - ヱ", 46 : "WO - ヲ", 47 : "N - ン"
}

print(indexToLabel[score2.argmax(axis=-1)[0]])


#le = preprocessing.LabelEncoder()
#print(score2)
#score2 = score2[0,:].astype(int)
#print(score2)


