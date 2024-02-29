import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

# Ganti 'nama_file.csv' dengan nama file CSV yang sesuai
file_path = '/kaggle/input/transjakarta/dfTransjakarta.csv'

# Baca file CSV
df = pd.read_csv(file_path)

# Tampilkan nama kolom
print("Nama Kolom dalam CSV:")
for column in df.columns:
    print(column)

    
# Tampilkan kolom-kolom dengan tipe data string
string_columns = df.select_dtypes(include='object').columns
print("Kolom-kolom dengan tipe data string:")
print(string_columns)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

# Ganti 'nama_file.csv' dengan nama file CSV yang sesuai
file_path = '/kaggle/input/transjakarta/dfTransjakarta.csv'

# Baca file CSV
df = pd.read_csv(file_path)

# Tangani nilai NaN dengan menggantinya menggunakan nilai rata-rata
# Tangani kolom numerik
numeric_columns = df.select_dtypes(include=['number']).columns
imputer_numeric = SimpleImputer(strategy='mean')
df[numeric_columns] = imputer_numeric.fit_transform(df[numeric_columns])

# Tangani kolom non-numerik
non_numeric_columns = df.select_dtypes(exclude=['number']).columns
imputer_non_numeric = SimpleImputer(strategy='most_frequent')  # Ganti dengan strategi yang sesuai
df[non_numeric_columns] = imputer_non_numeric.fit_transform(df[non_numeric_columns])

# Konversi kolom-kolom dengan tipe data string menjadi numerik
df['transID'] = df['transID'].astype('category').cat.codes
df['payCardBank'] = df['payCardBank'].astype('category').cat.codes
df['payCardName'] = df['payCardName'].astype('category').cat.codes
df['payCardSex'] = df['payCardSex'].astype('category').cat.codes
df['corridorID'] = df['corridorID'].astype('category').cat.codes
df['corridorName'] = df['corridorName'].astype('category').cat.codes
df['tapInStops'] = df['tapInStops'].astype('category').cat.codes
df['tapInStopsName'] = df['tapInStopsName'].astype('category').cat.codes
df['tapOutStops'] = df['tapOutStops'].astype('category').cat.codes
df['tapOutStopsName'] = df['tapOutStopsName'].astype('category').cat.codes

# Ubah format waktu menjadi datetime untuk memudahkan analisis waktu
df['tapInTime'] = pd.to_datetime(df['tapInTime'])
df['tapOutTime'] = pd.to_datetime(df['tapOutTime'])

# Ekstrak fitur waktu untuk model regresi linier
df['tapInHour'] = df['tapInTime'].dt.hour
df['tapInDay'] = df['tapInTime'].dt.day
df['tapInMonth'] = df['tapInTime'].dt.month
df['tapOutHour'] = df['tapOutTime'].dt.hour
df['tapOutDay'] = df['tapOutTime'].dt.day
df['tapOutMonth'] = df['tapOutTime'].dt.month

# Pilih fitur dan target
features = df.drop(['payAmount', 'tapInTime', 'tapOutTime'], axis=1)  # Exclude target column and time-related columns
target = df['payAmount']

# Bagi dataset menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Model Regresi Linier
reg_model = LinearRegression()
reg_model.fit(X_train, y_train)
reg_predictions = reg_model.predict(X_test)
reg_mse = mean_squared_error(y_test, reg_predictions)
print(f'MSE Regresi Linier: {reg_mse}')

# Model Time Series (ARIMA)
df.set_index('tapInTime', inplace=True)
arima_model = ARIMA(df['payAmount'], order=(1, 1, 1))  # Sesuaikan parameter p, d, q sesuai kebutuhan
arima_fit = arima_model.fit()
arima_predictions = arima_fit.predict(start=len(df), end=len(df) + len(X_test) - 1)
arima_mse = mean_squared_error(y_test, arima_predictions)
print(f'MSE ARIMA: {arima_mse}')

# Visualisasi hasil
plt.figure(figsize=(12, 6))

# Plot hasil regresi
plt.subplot(2, 2, 1)
plt.scatter(X_test['tapInHour'], y_test, color='black', label='Actual')
plt.scatter(X_test['tapInHour'], reg_predictions, color='blue', label='Regression Prediction')
plt.xlabel('Tap In Hour')
plt.ylabel('Pay Amount')
plt.title('Regression Results')
plt.legend()

# Plot hasil ARIMA
plt.subplot(2, 2, 2)
plt.plot(arima_predictions, color='red', label='ARIMA Prediction')
plt.plot(y_test.reset_index(drop=True), color='black', label='Actual')
plt.xlabel('Sample Index')
plt.ylabel('Pay Amount')
plt.title('ARIMA Results')
plt.legend()

# Perbandingan kedua model
plt.subplot(2, 2, 3)
plt.scatter(X_test['tapInHour'], y_test, color='black', label='Actual')
plt.scatter(X_test['tapInHour'], reg_predictions, color='blue', label='Regression Prediction')
plt.plot(arima_predictions, color='red', label='ARIMA Prediction')
plt.xlabel('Tap In Hour')
plt.ylabel('Pay Amount')
plt

# Define a function to visualize the results
def visualize_results(tap_in_hour, model_choice):
    plt.figure(figsize=(8, 4))

    if model_choice == 'Regression':
        predictions = reg_model.predict(X_test[X_test['tapInHour'] == tap_in_hour])
        plt.scatter(X_test[X_test['tapInHour'] == tap_in_hour]['tapInHour'], y_test[X_test['tapInHour'] == tap_in_hour], color='black', label='Actual')
        plt.scatter(X_test[X_test['tapInHour'] == tap_in_hour]['tapInHour'], predictions, color='blue', label='Regression Prediction')
        plt.xlabel('Tap In Hour')
        plt.ylabel('Pay Amount')
        plt.title('Regression Results')
        plt.legend()
    elif model_choice == 'ARIMA':
        predictions = arima_fit.predict(start=len(df), end=len(df) + len(X_test) - 1)
        plt.plot(predictions, color='red', label='ARIMA Prediction')
        plt.plot(y_test.reset_index(drop=True), color='black', label='Actual')
        plt.xlabel('Sample Index')
        plt.ylabel('Pay Amount')
        plt.title('ARIMA Results')
        plt.legend()

# Create interactive widgets
tap_in_hour_widget = widgets.IntSlider(min=df['tapInHour'].min(), max=df['tapInHour'].max(), step=1, description='Tap In Hour')
model_choice_widget = widgets.Dropdown(options=['Regression', 'ARIMA'], value='Regression', description='Model Choice')

# Create interactive dashboard
interact(visualize_results, tap_in_hour=tap_in_hour_widget, model_choice=model_choice_widget)
