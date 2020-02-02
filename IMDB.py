import numpy as np
from keras.datasets import imdb
import pandas as pd
# save np.load

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old



data_train=pd.DataFrame(train_data)
data_test=pd.DataFrame(test_data)
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

df_train = pd.DataFrame(mlb.fit_transform(data_train[0]),columns=mlb.classes_, index=data_train.index)

a=mlb.classes_

mlb = MultiLabelBinarizer(classes=a)

df_test = pd.DataFrame(mlb.fit_transform(data_test[0]), index=data_train.index)  


from keras.layers import Dense, Activation ,Dropout
from keras.models import Sequential 
from keras import regularizers
model=Sequential()
model.add(Dense(64,input_shape=(9998,),kernel_regularizer=regularizers.l2(0.005)))
model.add(Activation("tanh"))

model.add(Dropout(0.5))

model.add(Dense(128,kernel_regularizer=regularizers.l2(0.005)))
model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(64,kernel_regularizer=regularizers.l2(0.005)))
model.add(Activation("relu"))

model.add(Dropout(0.5))

model.add(Dense(1))
model.add(Activation("sigmoid"))

model.compile(optimizer='sgd',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history=model.fit(df_train,train_labels,epochs=20,batch_size=64,verbose=1,validation_data=(df_test,test_labels))



import matplotlib.pyplot as plt
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
plt.clf() #Clears the figure
acc = history_dict['acc']
val_acc = history_dict['val_acc']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()