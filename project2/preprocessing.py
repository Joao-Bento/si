import pandas as pd
import os
import numpy as np

class merge:
    def __init__(self, data_path, data_type):
        self.data_path = data_path
        self.data_type = data_type

        spheres_csv_file = os.path.join(data_path, 'spheres.csv')
        spheres = pd.read_csv(spheres_csv_file)

        data_csv_file = os.path.join(data_path, data_type)
        data = pd.read_csv(data_csv_file)
        
        spheres = spheres.iloc[::2]
        spheres['id'] = spheres['id'].astype(int)
        spheres['spawn_time'] = spheres['spawn_time'].astype(float)
        spheres['destruction_time'] = spheres['destruction_time'].astype(float)
        
        acquisition_start = spheres[spheres['id'] == 1].index[1]
        spheres = spheres[spheres.index >= acquisition_start]
        if len(spheres) > 20:
            acquisition_end = spheres[spheres['id'] > 20].index[0]
            spheres = spheres[spheres.index < acquisition_end]
        self.merger = -np.ones((len(spheres),3))
        
        ti = data['timestamp'][0]
        data['time_seconds'] = (data['timestamp'] - ti) / 10**7

        for id in spheres['id']:
            if id > 20 or id < 1: 
                break
            exc_duration = spheres[spheres['id'] == id]['destruction_time'].values[0] - spheres[spheres['id'] == id]['spawn_time'].values[0]
            start_index = data[data['id'] == id].index[0]
            start_time = data.loc[start_index, 'time_seconds']  
            end_time = start_time + exc_duration
            end_index = data[(data['time_seconds'] >= end_time)].index[0]
            self.merger[id-1] = [id, start_index, end_index]

    
    

