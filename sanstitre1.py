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




from keras.datasets import boston_housing
from sklearn.preprocessing import MultiLabelBinarizer

(train_data , train_labels),(test_data , test_labels)=boston_housing.load_data()



from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
df_train=scaler.fit_transform(train_data)
df_test=scaler.fit_transform(test_data)



from keras.layers import Dense, Activation
from keras.models import Sequential 

model=Sequential()
model.add(Dense(1024,input_shape=(13,)))
model.add(Activation("relu"))

#model.add(Dense(1024))
#model.add(Activation("relu"))



model.add(Dense(1))
model.add(Activation("relu"))

model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['mae'],)


history=model.fit(df_train,train_labels,epochs=50,batch_size=32,verbose=1,validation_data=(df_test,test_labels))







from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

seed = 7

noeuds=[128,512,1024]
b_s=[i for i in range(16,25)]
epoch=[10,15,20,25,30]

kfold = KFold(n_splits=4, shuffle=True, random_state=42)
cvscores = {}
for n in noeuds:
    for batch in b_s:
        for e in epoch:
            l=[]
            for train, test in kfold.split(df_train):
                x_train=df_train[train]
                x_test=df_train[test]
                y_train=train_labels[train]
                y_test=train_labels[test]
                model = Sequential()
                model.add(Dense(n, input_shape=(13,), activation='relu'))
                #model.add(Dense(8, activation='relu'))
                model.add(Dense(1, activation='relu'))
                # Compile model
                model.compile(optimizer='rmsprop',loss='mean_squared_error',metrics=['mae'],)
                # Fit the model
                model.fit(x_train,y_train, epochs=e, batch_size=batch,validation_data=(x_test,y_test),verbose=0)
                # evaluate the model
                scores = model.evaluate(x_test,y_test)
                l.append(scores)
                
            print(np.mean(l))
            cvscores[np.mean(l)]=(n,batch,e)
    




