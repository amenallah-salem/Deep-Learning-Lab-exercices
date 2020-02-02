from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
import numpy as np 
"""    XOR    """
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]])

model = Sequential()
model.add(Dense(8, input_dim=2))
model.add(Activation('tanh'))
model.add(Dense(1))
model.add(Activation('sigmoid'))

sgd = SGD(lr=0.3)
model.compile(loss='binary_crossentropy', optimizer=sgd,metrics=['accuracy'])

model.fit(X, y, batch_size=1, nb_epoch=50,)

print(model.predict(X))



"""    MNIST     """

from keras.datasets import mnist
(train_images, train_labels), (test_images,test_labels) = mnist.load_data()