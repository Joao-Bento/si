import pandas as pd
import numpy as np
import preprocessing

class run:
    def __init__(self, path, sensor):

        self.path = path
        self.sensor = sensor


        if self.sensor == 'acc':
            imu_path = f'{self.path}/imu_acc_data.csv'
            imu_data ='imu_acc_data.csv'
        elif self.sensor == 'gyro':
            imu_path = f'{self.path}/imu_gyro_data.csv'
            imu_data ='imu_gyro_data.csv'
        elif self.sensor == 'mag':
            imu_path = f'{self.path}/imu_mag_data.csv'
            imu_data ='imu_mag_data.csv'
            
        #self.exc_df = pd.read_csv(exc_path)
        self.imu_df = pd.read_csv(imu_path)
        exercises = preprocessing.merge(self.path, imu_data)
        self.merger = exercises.merger
        
        self.process_data()
        self.features = {
            "id": [],
            f"x_mean_{self.sensor}": [],
            f"x_var_{self.sensor}": [],
            f"x_std_{self.sensor}": [],
            f"x_max_{self.sensor}": [],
            f"x_min_{self.sensor}": [],
            f"y_mean_{self.sensor}": [],
            f"y_var_{self.sensor}": [],
            f"y_std_{self.sensor}": [],
            f"y_max_{self.sensor}": [],
            f"y_min_{self.sensor}": [],
            f"z_mean_{self.sensor}": [],
            f"z_var_{self.sensor}": [],
            f"z_std_{self.sensor}": [],
            f"z_max_{self.sensor}": [],
            f"z_min_{self.sensor}": []
        }
        
        self.feature_extraction()
        
        
        
    def process_data(self):
        ti_imu = self.imu_df['timestamp'][0]
        self.imu_df['time_seconds'] = (self.imu_df['timestamp'] - ti_imu) / 10**7

        # Find discontinuities in timestamp
        # time_diff = self.imu_df['time_seconds'].diff()
        # discontinuities = self.imu_df[(time_diff > 1) | (time_diff < -1)]

        # for i in discontinuities.index:
        #     offset = self.imu_df['time_seconds'].loc[i] - self.imu_df['time_seconds'].loc[i - 1]
        #     if offset > 1:
        #         self.imu_df.loc[i:, 'time_seconds'] -= offset
        #     else:
        #         self.imu_df.loc[i:, 'time_seconds'] += abs(offset)
                
        # Drop irrelevant columns
        self.imu_df = self.imu_df.drop(columns=['timestamp', 'count', 'sensor_ticks', 'soc_ticks', 'temperature'])

        # Transform data from string to numpy array
        self.imu_df['x'] = self.imu_df['x'].apply(lambda v: np.fromstring(v.strip("[]"), sep=" "))
        self.imu_df['y'] = self.imu_df['y'].apply(lambda v: np.fromstring(v.strip("[]"), sep=" "))
        self.imu_df['z'] = self.imu_df['z'].apply(lambda v: np.fromstring(v.strip("[]"), sep=" "))

        self.imu_df['x'] = self.imu_df.apply(lambda row: np.mean(row['x']), axis=1)
        self.imu_df['y'] = self.imu_df.apply(lambda row: np.mean(row['y']), axis=1)
        self.imu_df['z'] = self.imu_df.apply(lambda row: np.mean(row['z']), axis=1)

    def stats_features(self, df, col):
        mean = df[col].mean()
        var = df[col].var()
        std = df[col].std()
        max_val = df[col].max()
        min_val = df[col].min()
        return [mean, var, std, max_val, min_val]
    
    def feature_extraction(self):
        for id in self.merger[:,0]:
            id = int(id)
            spawn_index = int(self.merger[id-1][1])
            destruction_index = int(self.merger[id-1][2])
            #mask = (self.eet_df['time_seconds'] >= spawn_time) & (self.eet_df['time_seconds'] <= destruction_time)
            instance = self.imu_df.loc[spawn_index:destruction_index+1]
            x_features = self.stats_features(instance, 'x')
            y_features = self.stats_features(instance, 'y')
            z_features = self.stats_features(instance, 'z')
            self.features['id'].append(id)
            self.features[f"x_mean_{self.sensor}"].append(x_features[0])
            self.features[f"x_var_{self.sensor}"].append(x_features[1])
            self.features[f"x_std_{self.sensor}"].append(x_features[2])
            self.features[f"x_max_{self.sensor}"].append(x_features[3])
            self.features[f"x_min_{self.sensor}"].append(x_features[4])
            self.features[f"y_mean_{self.sensor}"].append(y_features[0])
            self.features[f"y_var_{self.sensor}"].append(y_features[1])
            self.features[f"y_std_{self.sensor}"].append(y_features[2])
            self.features[f"y_max_{self.sensor}"].append(y_features[3])
            self.features[f"y_min_{self.sensor}"].append(y_features[4])
            self.features[f"z_mean_{self.sensor}"].append(z_features[0])
            self.features[f"z_var_{self.sensor}"].append(z_features[1])
            self.features[f"z_std_{self.sensor}"].append(z_features[2])
            self.features[f"z_max_{self.sensor}"].append(z_features[3])
            self.features[f"z_min_{self.sensor}"].append(z_features[4])
    
