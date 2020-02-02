import numpy as np
from keras.datasets import reuters
import pandas as pd
# save np.load

np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# call load_data with allow_pickle implicitly set to true
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# restore np.load for future normal usage
np.load = np_load_old




from keras.datasets import reuters
from sklearn.preprocessing import MultiLabelBinarizer

(train_data , train_labels),(test_data , test_labels)=reuters.load_data(num_words=10000)


data_train=pd.DataFrame(train_data)
mlb = MultiLabelBinarizer()
df_train = pd.DataFrame(mlb.fit_transform(data_train[0]),columns=mlb.classes_, index=data_train.index)
a=mlb.classes_

data_test=pd.DataFrame(test_data)
mlb = MultiLabelBinarizer()
df_test = pd.DataFrame(mlb.fit_transform(data_test[0]),columns=mlb.classes_, index=data_test.index)
b=mlb.classes_  

c=set(a).union(set(b))

mlb = MultiLabelBinarizer(classes=np.array(list(c)))
df_train = pd.DataFrame(mlb.fit_transform(data_train[0]), index=data_train.index)
df_test = pd.DataFrame(mlb.fit_transform(data_test[0]), index=data_test.index)

from keras.utils import to_categorical
y_train=to_categorical(train_labels)
y_test=to_categorical(test_labels)




from keras.layers import Dense, Activation
from keras.models import Sequential 

model=Sequential()
model.add(Dense(8,input_shape=(9998,)))
model.add(Activation("tanh"))

model.add(Dense(4))
model.add(Activation("tanh"))



model.add(Dense(46))
model.add(Activation("softmax"))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'],)


history=model.fit(df_train,y_train,epochs=10,batch_size=128,verbose=1,validation_data=(df_test,y_test))





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



