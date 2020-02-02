import numpy as np
from keras.datasets import cifar10
import pandas as pd
# save np.load

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = cifar10.load_data()

# restore np.load for future normal usage
np.load = np_load_old

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten , MaxPooling2D ,Dropout,add,Input


X_train=train_data.astype("float32")/255
X_test=test_data.astype("float32")/255

Y_train=to_categorical(train_labels,num_classes=10)
Y_test=to_categorical(test_labels,num_classes=10)

"""
model= Sequential()  

model.add(Conv2D(128, (3, 3), input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(128, (3, 3), activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dense(10, activation='softmax'))

sgd = SGD(lr=0.3)
model.compile(loss='categorical_crossentropy', optimizer="rmsprop",metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=5,batch_size=128,validation_data=(X_test,Y_test))
"""
import keras

#Model= Sequential()
inputt=Input(shape=(32, 32, 3))
x=inputt
x=Conv2D(32, (3, 3), activation='relu',padding="same")(x)
x=MaxPooling2D(pool_size=(2, 2))(x)
for i in range(5):
    indet=x
    x=Conv2D(32, (3, 3), activation='relu',padding="same")(x)
    #x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Conv2D(32, (3, 3), activation='relu',padding="same")(x)
    #x=MaxPooling2D(pool_size=(2, 2))(x)
    x=add([indet,x])
    
faltten=Flatten()(x)
dense=Dense(10, activation='softmax')(faltten)
model=keras.models.Model(inputs=inputt,outputs=dense)

model.compile(loss='categorical_crossentropy', optimizer="rmsprop",metrics=['accuracy'])
model.fit(X_train, Y_train, nb_epoch=5,batch_size=128,validation_data=(X_test,Y_test))







