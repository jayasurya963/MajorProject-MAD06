# Importing necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from tensorflow import keras
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv('/content/drive/MyDrive/MLLab/major/SpaceShipTracking.csv')

print(data.head)
print(data.tail)
print(data.describe)

data['Target'] = (data['Ship_ID'].shift(-1) > data['Ship_ID']).astype(int)

# Drop rows with NaN values
data.dropna(inplace=True)

# Gathering the Information
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data[['Ship_ID', 'Age', 'Ship_class', 'Ship_Speed']])
print(scaled_data)

# Create features and target variable
X = scaled_data[:, :-1]
y = data['Target'].values

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizing the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_reg_pred = log_reg.predict(X_test)
log_reg_accuracy = accuracy_score(y_test, log_reg_pred)
print(log_reg_accuracy)

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_pred = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)
print(dt_accuracy)

# Random Forest Classifier
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_pred = rf_classifier.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(rf_accuracy)

# K-Nearest Neighbors Classifier
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_pred = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
print(knn_accuracy)


# ARIMA Model
# Assuming you have time series data for ARIMA
# Splitting data for ARIMA
train_data = data[:int(0.8*(len(data)))]
test_data = data[int(0.8*(len(data))):]

# LSTM Model
# Assuming you have sequential data for LSTM
# Reshape the data
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], 1)))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dense(units=1))

# Compile the model
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
lstm_model.fit(X_train_lstm, y_train, epochs=100, batch_size=32)

# Predictions
lstm_predictions = lstm_model.predict(X_test_lstm)

# Visualization (assuming you want to visualize predictions)
plt.plot(test_data.index, test_data['Ship_class'], label='Actual')
plt.plot(test_data.index, lstm_predictions, label='LSTM')
plt.legend()
plt.show()


# Visualization (assuming you want to visualize predictions)
plt.plot(test_data.index, test_data['Ship_Speed'], label='Actual')
plt.plot(test_data.index, lstm_predictions, label='LSTM')
plt.legend()
plt.show()

# Visualization (assuming you want to visualize predictions)
plt.plot(test_data.index, test_data['Ship_Fuel_Level'], label='Actual')
plt.plot(test_data.index, lstm_predictions, label='LSTM')
plt.legend()
plt.show()
