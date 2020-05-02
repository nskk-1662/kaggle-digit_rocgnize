import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
#convert_to_one-hot-encoding
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

from digit_model import cnn_model
from readData import train,test,x_train,y_train

# plt.subplot(231)
# plt.imshow(np.reshape(np.array(x_train.iloc[0]),(28,28)), cmap=plt.get_cmap('gray'))
# plt.subplot(232)
# plt.imshow(np.reshape(np.array(x_train.iloc[1]),(28,28)), cmap=plt.get_cmap('gray'))
# plt.subplot(233)
# plt.imshow(np.reshape(np.array(x_train.iloc[2]),(28,28)), cmap=plt.get_cmap('gray'))
# plt.subplot(234)
# plt.imshow(np.reshape(np.array(x_train.iloc[3]),(28,28)), cmap=plt.get_cmap('gray'))
# plt.subplot(235)
# plt.imshow(np.reshape(np.array(x_train.iloc[4]),(28,28)), cmap=plt.get_cmap('gray'))
# plt.subplot(236)
# plt.imshow(np.reshape(np.array(x_train.iloc[5]),(28,28)), cmap=plt.get_cmap('gray'))
# plt.show()

#データ処理
x_train=x_train/255
test=test/255

x_train = x_train.values.reshape(-1,28,28,1)
test = test.values.reshape(-1,28,28,1)

y_train=to_categorical(y_train,num_classes=10)
print(x_train.shape,y_train.shape,test.shape)

num_classes = y_train.shape[1]
num_pixels = x_train.shape[1]

x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.20, random_state=0)

model=cnn_model()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3, verbose=1, factor=0.5, min_lr=0.00001)
epochs = 1 #make 30 for best results(99.3% Accuracy)
batch_size = 200

#データ水増し
datagen = ImageDataGenerator()
datagen.fit(x_train)

#training
model.fit_generator(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0], callbacks=[learning_rate_reduction])

model.save_weights("model.h5")

#pred
submit.Label =model.predict_classes(test)
submit.head()
submit.to_csv('submit.csv',index=False)
