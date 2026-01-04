# Live Stock Market Forecasting
# Forecasting stock price for next 30 days using LSTM

# Yahoo finance as data source
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# See the yahoo finance ticker for your stock symbol
stock_symbol = 'URTH'

# Last 5 years data with interval of 1 day
data = yf.download(tickers=stock_symbol, period='5y', interval='1d')

print(f"Total records: {len(data)}")
print("\nFirst 5 rows:")
print(data.head())
print("\nLast 5 rows:")
print(data.tail())

# Extract Open prices
opn = data[['Open']]
ds = opn.values

# Using MinMaxScaler for normalizing data between 0 & 1
normalizer = MinMaxScaler(feature_range=(0, 1))
ds_scaled = normalizer.fit_transform(np.array(ds).reshape(-1, 1))

# Defining test and train data sizes
train_size = int(len(ds_scaled) * 0.70)
test_size = len(ds_scaled) - train_size

print(f"\nTrain size: {train_size}, Test size: {test_size}")

# Splitting data between train and test
ds_train, ds_test = ds_scaled[0:train_size, :], ds_scaled[train_size:len(ds_scaled), :1]


# Creating dataset in time series for LSTM model
# X[100,120,140,160,180] : Y[200]
def create_ds(dataset, step):
    Xtrain, Ytrain = [], []
    for i in range(len(dataset) - step - 1):
        a = dataset[i:(i + step), 0]
        Xtrain.append(a)
        Ytrain.append(dataset[i + step, 0])
    return np.array(Xtrain), np.array(Ytrain)


# Taking 100 days price as one record for training
time_stamp = 100
X_train, y_train = create_ds(ds_train, time_stamp)
X_test, y_test = create_ds(ds_test, time_stamp)

print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

# Reshaping data to fit into LSTM model
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Creating LSTM model using keras
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=50, return_sequences=True))
model.add(LSTM(units=50))
model.add(Dense(units=1, activation='linear'))
model.summary()

# Training model with adam optimizer and mean squared error loss function
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64)

# Plotting loss
loss = model.history.history['loss']
plt.figure(figsize=(10, 6))
plt.plot(loss)
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('model_loss.png')
print("\nModel loss plot saved as 'model_loss.png'")

# Predicting on train and test data
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Inverse transform to get actual value
train_predict = normalizer.inverse_transform(train_predict)
test_predict = normalizer.inverse_transform(test_predict)

# Combining the predicted data
test = np.vstack((train_predict, test_predict))

# Plotting comparison
plt.figure(figsize=(12, 6))
plt.plot(normalizer.inverse_transform(ds_scaled), label='Actual Data')
plt.plot(test, label='Predicted Data')
plt.title(f'{stock_symbol} - Actual vs Predicted')
plt.ylabel('Price')
plt.xlabel('Time')
plt.legend()
plt.savefig('train_test_prediction.png')
print("Train/Test prediction plot saved as 'train_test_prediction.png'")

# Getting the last 100 days records
fut_inp = ds_test[len(ds_test) - time_stamp:]
fut_inp = fut_inp.reshape(1, -1)
tmp_inp = list(fut_inp)
tmp_inp = tmp_inp[0].tolist()

# Predicting next 30 days price using the current data
# It will predict in sliding window manner (algorithm) with stride 1
lst_output = []
n_steps = 100
i = 0

print("\nPredicting next 30 days...")
while(i < 30):
    if(len(tmp_inp) > 100):
        fut_inp = np.array(tmp_inp[1:])
        fut_inp = fut_inp.reshape(1, -1)
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        tmp_inp = tmp_inp[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        fut_inp = fut_inp.reshape((1, n_steps, 1))
        yhat = model.predict(fut_inp, verbose=0)
        tmp_inp.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

print("Prediction complete!")

# Creating a dummy plane to plot graph one after another
plot_new = np.arange(1, 101)
plot_pred = np.arange(101, 131)

plt.figure(figsize=(10, 6))
plt.plot(plot_new, normalizer.inverse_transform(ds_scaled[len(ds_scaled) - 100:]))
plt.plot(plot_pred, normalizer.inverse_transform(lst_output))
plt.title(f'{stock_symbol} - Last 100 days and Next 30 days prediction')
plt.ylabel('Price')
plt.xlabel('Time')
plt.savefig('next_30days_prediction.png')
print("Next 30 days prediction plot saved as 'next_30days_prediction.png'")

# Creating final data for plotting
ds_new = ds_scaled.tolist()
ds_new.extend(lst_output)
final_graph = normalizer.inverse_transform(ds_new).tolist()

# Plotting final results with predicted value after 30 Days
plt.figure(figsize=(14, 7))
plt.plot(final_graph)
plt.ylabel("Price")
plt.xlabel("Time")
plt.title(f"{stock_symbol} prediction of next month open")
plt.axhline(y=final_graph[len(final_graph) - 1], color='red', linestyle=':',
            label=f'NEXT 30D: {round(float(*final_graph[len(final_graph) - 1]), 2)}')
plt.legend()
plt.savefig('final_prediction.png')
print(f"Final prediction plot saved as 'final_prediction.png'")
print(f"\nPredicted price after 30 days: ${round(float(*final_graph[len(final_graph) - 1]), 2)}")

plt.show()
