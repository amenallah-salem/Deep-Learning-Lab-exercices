import numpy as np
from keras.datasets import cifar10
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.layers import Conv2D, Flatten , MaxPooling2D ,Dropout,add,Input ,LSTM


data=pd.read_csv("sinwave.csv")

d=data.values.tolist()

dataframe=[]
j=0
for i in range(len(d)-51):
    l=[]
    for k in range(j,j+51):
        l.append(d[k][0])
    l=np.array(l)
    dataframe.append(l)
    j=j+1
    
dataframe=np.array(dataframe)
y=[ i[len(i)-1]    for i in dataframe]
y=np.array(y)
dataframe=dataframe.reshape(4949,51,1)



from sklearn.model_selection import train_test_split
validation_size = 0.30
X_train, X_test , Y_train , Y_test= train_test_split(dataframe,y, test_size=validation_size)



model_lstm = Sequential()
model_lstm.add(LSTM(50,return_sequences=True,dropout=0.2))
#model_lstm(Dropout(0.2))
model_lstm.add(LSTM(100,return_sequences=False,dropout=0.2))
#model_lstm(Dropout(0.2))
#model_lstm.add(Dense(256, activation = 'relu'))

model_lstm.add(Dense(1))#, activation = 'softmax'))
model_lstm.compile(loss='mean_squared_error',optimizer='adam',metrics=['mae']
)


model_lstm.fit(X_train,Y_train,epochs=10,verbose=1,validation_data=(X_test,Y_test))
