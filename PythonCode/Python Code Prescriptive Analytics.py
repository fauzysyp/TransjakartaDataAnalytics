!pip install -U scikit-fuzzy
!pip install numpy
!pip install pandas

#Install module scikitfuzzy, numpy, dan pandas
import pandas as pd
import numpy as np 
import datetime
import skfuzzy as fuzz 
from skfuzzy import control as ctrl

# Load  dataset

df = pd.read_csv('/kaggle/input/transjkt/dfTransjakarta.csv')
print(df)

def checkNA(data):
    if data.isna().sum().any():
        print('There are null values, we will remove them...')
        data.dropna(inplace=True)
    else:
        pass
        
checkNA(df)

df.isna().sum()

df.info()

df['tapInTime'] = pd.to_datetime(df['tapInTime'])
df['tapInHour'] = df['tapInTime'].dt.hour
df['tapOutTime'] = pd.to_datetime(df['tapOutTime'])
df['runtime'] = (df['tapOutTime'] - df['tapInTime'])
#mengubah ke menit
df['travel_time'] = pd.to_timedelta(df['runtime']).dt.total_seconds() / 60
print(df["travel_time"])
print(df["tapInHour"])

penumpang = df.groupby('corridorID')['corridorID'].count().reset_index(name='jumlah_penumpang')
for index, row in penumpang.iterrows():
    print(f"corridorID: {row['corridorID']}, JumlahPenumpang: {row['jumlah_penumpang']}")

print(penumpang)

corridorID = '10'  # Replace with the actual corridorID
jumlah_penumpang = penumpang.loc[penumpang['corridorID'] == corridorID, 'jumlah_penumpang'].values[0]
print("Jumlah penumpang for corridor", corridorID, ":", jumlah_penumpang)

travel_time = ctrl.Antecedent(np.arange(0, 200, 1), 'travel_time') #format dalam menit
tapInHour = ctrl.Antecedent(np.arange(0, 24, 1), 'tapInHour') #format 24 jam
jumlah_penumpang = ctrl.Antecedent(np.arange(0, 500, 10), 'jumlah_penumpang')

bus_frequency = ctrl.Consequent(np.arange(0, 11, 1), 'bus_frequency')

# Define membership functions (use descriptive names)
travel_time['good'] = fuzz.trapmf(travel_time.universe, [0, 0, 15, 30])
travel_time['average'] = fuzz.trimf(travel_time.universe, [15, 45, 75])
travel_time['poor'] = fuzz.trapmf(travel_time.universe, [60, 90, 200, 201])
travel_time.view()

tapInHour['peak_morning'] = fuzz.trapmf(tapInHour.universe, [4, 7, 9, 11])
tapInHour['afternoon'] = fuzz.trimf(tapInHour.universe, [10, 15, 18])
tapInHour['peak_evening'] = fuzz.trapmf(tapInHour.universe, [16, 20, 23, 24])
tapInHour.view()

jumlah_penumpang['low'] = fuzz.trapmf(jumlah_penumpang.universe, [0, 50, 100, 150])
jumlah_penumpang['average'] = fuzz.trimf(jumlah_penumpang.universe, [100, 250, 400])
jumlah_penumpang['high'] = fuzz.trapmf(jumlah_penumpang.universe, [300, 400, 500, 501])
jumlah_penumpang.view()

bus_frequency['good'] = fuzz.trapmf(bus_frequency.universe, [0, 2, 3, 5])
bus_frequency['average'] = fuzz.trimf(bus_frequency.universe, [3, 5, 7])
bus_frequency['poor'] = fuzz.trimf(bus_frequency.universe, [6, 8, 10])
bus_frequency.view()

# Define fuzzy rules
rule1 = ctrl.Rule(travel_time['poor'] & tapInHour['peak_morning'] & jumlah_penumpang['low'], bus_frequency['average'])
rule2 = ctrl.Rule(travel_time['poor'] & tapInHour['afternoon'] & jumlah_penumpang['average'], bus_frequency['average'])
rule3 = ctrl.Rule(travel_time['poor'] & tapInHour['peak_evening'] & jumlah_penumpang['high'], bus_frequency['poor'])
rule4 = ctrl.Rule(travel_time['poor'] & tapInHour['peak_morning'] & jumlah_penumpang['average'], bus_frequency['poor'])
rule5 = ctrl.Rule(travel_time['poor'] & tapInHour['afternoon'] & jumlah_penumpang['high'], bus_frequency['poor'])
rule6 = ctrl.Rule(travel_time['poor'] & tapInHour['peak_evening'] & jumlah_penumpang['low'], bus_frequency['average'])
rule7 = ctrl.Rule(travel_time['poor'] & tapInHour['peak_morning'] & jumlah_penumpang['high'], bus_frequency['poor'])
rule8 = ctrl.Rule(travel_time['poor'] & tapInHour['afternoon'] & jumlah_penumpang['low'], bus_frequency['good'])
rule9 = ctrl.Rule(travel_time['poor'] & tapInHour['peak_evening'] & jumlah_penumpang['average'], bus_frequency['poor'])

rule10 = ctrl.Rule(travel_time['average'] & tapInHour['peak_morning'] & jumlah_penumpang['low'], bus_frequency['poor'])
rule11 = ctrl.Rule(travel_time['average'] & tapInHour['afternoon'] & jumlah_penumpang['average'], bus_frequency['average'])
rule12 = ctrl.Rule(travel_time['average'] & tapInHour['peak_evening'] & jumlah_penumpang['high'], bus_frequency['poor'])
rule13 = ctrl.Rule(travel_time['average'] & tapInHour['peak_morning'] & jumlah_penumpang['average'], bus_frequency['good'])
rule14 = ctrl.Rule(travel_time['average'] & tapInHour['afternoon'] & jumlah_penumpang['high'], bus_frequency['average'])
rule15 = ctrl.Rule(travel_time['average'] & tapInHour['peak_evening'] & jumlah_penumpang['low'], bus_frequency['average'])
rule16 = ctrl.Rule(travel_time['average'] & tapInHour['peak_morning'] & jumlah_penumpang['high'], bus_frequency['average'])
rule17 = ctrl.Rule(travel_time['average'] & tapInHour['afternoon'] & jumlah_penumpang['low'], bus_frequency['average'])
rule18 = ctrl.Rule(travel_time['average'] & tapInHour['peak_evening'] & jumlah_penumpang['average'], bus_frequency['good'])

rule19 = ctrl.Rule(travel_time['good'] & tapInHour['peak_morning'] & jumlah_penumpang['low'], bus_frequency['poor'])
rule20 = ctrl.Rule(travel_time['good'] & tapInHour['afternoon'] & jumlah_penumpang['average'], bus_frequency['average'])
rule21 = ctrl.Rule(travel_time['good'] & tapInHour['peak_evening'] & jumlah_penumpang['high'], bus_frequency['good'])
rule22 = ctrl.Rule(travel_time['good'] & tapInHour['peak_morning'] & jumlah_penumpang['average'], bus_frequency['average'])
rule23 = ctrl.Rule(travel_time['good'] & tapInHour['afternoon'] & jumlah_penumpang['high'], bus_frequency['good'])
rule24 = ctrl.Rule(travel_time['good'] & tapInHour['peak_evening'] & jumlah_penumpang['low'], bus_frequency['poor'])
rule25 = ctrl.Rule(travel_time['good'] & tapInHour['peak_morning'] & jumlah_penumpang['high'], bus_frequency['good'])
rule26 = ctrl.Rule(travel_time['good'] & tapInHour['afternoon'] & jumlah_penumpang['low'], bus_frequency['poor'])
rule27 = ctrl.Rule(travel_time['good'] & tapInHour['peak_evening'] & jumlah_penumpang['average'], bus_frequency['average'])


# Create a control system and simulate
bus_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9, rule10, rule11, rule12, rule13, rule14, rule15,rule16,rule17,rule18,rule19,rule20,rule21,rule22,rule23,rule24,rule25,rule26,rule27])
bus_simulation = ctrl.ControlSystemSimulation(bus_ctrl)

travel_time = input("Masukkan lama kamu berkendara dalam menit: ")
bus_simulation.input['travel_time'] = int(travel_time)

tapInHour = input("Masukkan Jam Berapa: ")
bus_simulation.input['tapInHour'] = int(tapInHour)

corridorID = input("Masukkan corridorID: ")
jumlah_penumpang = penumpang.loc[penumpang['corridorID'] == corridorID, 'jumlah_penumpang'].values[0]
bus_simulation.input['jumlah_penumpang'] = jumlah_penumpang

bus_simulation.compute()

print("Bus frequency:", round(bus_simulation.output['bus_frequency'], 2)) 
bus_frequency.view(sim=bus_simulation)

i = 0
total = 0
row = 37900

while i < row:
    bus_frequency = bus_simulation.output['bus_frequency']
    bus_simulation.input['travel_time'] = int(travel_time)
    bus_simulation.input['tapInHour'] = int(tapInHour)
    corridorID = input("Masukkan corridorID: ")
    jumlah_penumpang = penumpang.loc[penumpang['corridorID'] == corridorID, 'jumlah_penumpang'].values[0]
    bus_simulation.input['jumlah_penumpang'] = jumlah_penumpang
    bus_simulation.compute()

    if (bus_simulation.output['bus_frequency'] < 10) and (jumlah_penumpang == 'average'):
        result = "✔️"
        total += 1
    elif (bus_simulation.output['bus_frequency'] < 5) and (jumlah_penumpang == 'good'):
        result = "✔️"
        total += 1
    elif (bus_simulation.output['bus_frequency'] > 10) and (jumlah_penumpang == 'poor'):
        result = "✔️"
        total += 1
    else:
        result = "❌"

    print(i + 1, bus_simulation.output['bus_frequency'], bus_frequency, result, sep="   ")

    # Assuming 'data' is a DataFrame where you want to store results
    df.loc[i, 'Classification Result'] = bus_simulation.output['bus_frequency']
    df.to_csv("dfTransjakarta - New.csv", index=False)
    i += 1

print("")
print(total, "/", row)
print("Accuracy = ", total / row * 100, "%")
