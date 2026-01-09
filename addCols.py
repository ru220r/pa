import pandas as pd
from datetime import timedelta
import os


csv_path = r'c:\FH\PA\food_orders_new_delhi.csv'
df = pd.read_csv(csv_path)


df['Order Date and Time'] = pd.to_datetime(df['Order Date and Time'])
df['Delivery Date and Time'] = pd.to_datetime(df['Delivery Date and Time'])

df['Order Time'] = df['Order Date and Time'].dt.strftime('%H:%M:%S')

df['Delivery Duration'] = df['Delivery Date and Time'] - df['Order Date and Time']

df['Delivery Duration'] = df['Delivery Duration'].apply(
    lambda x: f"{int(x.total_seconds() // 3600):02d}:{int((x.total_seconds() % 3600) // 60):02d}:{int(x.total_seconds() % 60):02d}"
)

df.to_csv(csv_path, index=False)

print("✓ Neue Spalte 'Delivery Duration' hinzugefügt!")
print("\nErste 5 Zeilen:")
print(df.head())
