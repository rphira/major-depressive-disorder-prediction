import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('smartwatch_data.csv')

def clean_data(df):
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.drop_duplicates()
    df = df[(df['heart_rate'] > 40) & (df['heart_rate'] < 200)]
    return df

data = clean_data(data)


def feature_engineering(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    df['accel_mean'] = df['accelerometer'].rolling(window=10).mean()
    df['accel_std'] = df['accelerometer'].rolling(window=10).std()
 
    df['activity_level'] = np.where(df['accelerometer'] > threshold, 'active', 'inactive')
    
    return df

data = feature_engineering(data)

def transform_data(df):
    df = df.set_index('timestamp')
    df = df.resample('1T').mean()
    return df

data = transform_data(data)

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data.select_dtypes(include=[np.number]))

data_scaled_df = pd.DataFrame(data_scaled, columns=data.select_dtypes(include=[np.number]).columns)
data_final = pd.concat([data_scaled_df, data.select_dtypes(exclude=[np.number]).reset_index(drop=True)], axis=1)

train_data, test_data = train_test_split(data_final, test_size=0.2, random_state=42)

train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)
