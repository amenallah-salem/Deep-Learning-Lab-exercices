from keras.datasets import reuters
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
(train_data, train_labels), (test_data,test_labels)=reuters.load_data(num_words=10000)


df=pd.DataFrame(train_data)
df3=pd.DataFrame(test_data)
mlb = MultiLabelBinarizer()

df2 = pd.DataFrame(mlb.fit_transform(df[0]),columns=mlb.classes_, index=df.index)

a=mlb.classes_

mlb = MultiLabelBinarizer(classes=a)

df4 = pd.DataFrame(mlb.fit_transform(df3[0]), index=df.index)  

