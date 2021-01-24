import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

from rbf.rbflayer import RBFLayer, InitCentersKMeans

df = pd.read_csv(r'banknotes.txt', sep=',')
df.head()

X_train, X_test, y_train, y_test = train_test_split(df.drop('class', axis=1), df['class'])

rbflayer = RBFLayer(10, initializer=InitCentersKMeans(X_train), betas=2.0, input_shape=(1,))

outputlayer = Dense(2, activation="softmax")

input_ = Input(shape=X_train.shape[1:])

model = Sequential()
model.add(input_)
model.add(rbflayer)
model.add(outputlayer)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=100, epochs=1000)

model.evaluate(X_test, y_test)
y_pred = model.predict(X_test).argmax(axis=1)
conf = confusion_matrix(y_test, y_pred)

f, ax = plt.subplots(1, 1)

sns.heatmap(conf, square=True, annot=True, cbar=False, fmt='d', ax=ax)

plt.show()

class_report = classification_report(y_test, y_pred)

print(class_report)
