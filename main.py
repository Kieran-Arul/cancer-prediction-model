from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

df = pd.read_csv("cancer_data.csv")

# Get overview
# print(df.describe())

# Check for even balance of classes
sns.countplot(x='benign_0__mal_1', data=df)
# plt.show()

# Check correlation between each of the features
df.corr()['benign_0__mal_1'][:-1].sort_values().plot(kind='bar')
# plt.show()

# Features
X = df.drop('benign_0__mal_1', axis=1).values

# Target
y = df['benign_0__mal_1'].values

# TTS
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

# Scale data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training data shape to determine no. of units
# print(X_train.shape)

model = Sequential()

model.add(Dense(units=30, activation='relu'))

# Turn off 50% of neurons
model.add(Dropout(rate=0.5))

model.add(Dense(units=15, activation='relu'))
model.add(Dropout(rate=0.5))

model.add(Dense(units=1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam')

early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
model.fit(x=X_train, y=y_train, epochs=600, validation_data=(X_test, y_test), callbacks=[early_stop])

# Validation and training loss comparison
model_losses = pd.DataFrame(model.history.history)
model_losses.plot()
# plt.show()

# Obtain predictions on test data
predictions = (model.predict(X_test) > 0.5).astype("int32")

# Evaluate model (Approx 97% f1 score, 4 wrong classifications)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
