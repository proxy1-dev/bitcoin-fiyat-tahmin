import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input

SEED = 42 
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

df = pd.read_csv("bitcoin.csv")
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
series = df['Close'].values.reshape(-1, 1)

split = int(len(series) * 0.8)
train_data = series[:split]
test_data = series[split:]

sc = MinMaxScaler(feature_range=(0, 1))
scaled_train = sc.fit_transform(train_data)
scaled_test = sc.transform(test_data)

def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(scaled_train)
X_test, y_test = create_sequences(scaled_test)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

model = Sequential([
    Input(shape=(60, 1)),
    Bidirectional(LSTM(units=100, return_sequences=True)),
    Dropout(0.2),
    LSTM(units=100),
    Dropout(0.2),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], color='blue', label='Train Loss')
plt.plot(history.history['val_loss'], color='orange', label='Test Loss')
plt.title('Model Learning Performance')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()