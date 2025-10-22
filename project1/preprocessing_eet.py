import pandas as pd
import numpy as np

class run:
    def __init__(self, path):

        self.path = path

        exc_path = f'{self.path}/spheres.csv'
        eet_path = f'{self.path}/eet_data.csv'

        self.exc_df = pd.read_csv(exc_path)
        self.eet_df = pd.read_csv(eet_path)
        
        self.process_data()
        self.features = {
            "index": [],
            "blink_rate": [],
            "blink_duration_mean": [],
            "blink_duration_var": [],
            "blink_duration_std": []}
        self.feature_extraction()
        
    def process_data(self):
        ti_spawn_exc = self.exc_df['spawn_time'][0]
        ti_destruction_exc = self.exc_df['destruction_time'][0]
        ti_eet = self.eet_df['timestamp'][0]
        self.exc_df['spawn_time'] = self.exc_df['spawn_time'] - ti_spawn_exc
        self.exc_df['destruction_time'] = self.exc_df['destruction_time'] - ti_spawn_exc
        self.eet_df['time_seconds'] = (self.eet_df['timestamp'] - ti_eet) / 10**7
        
        import matplotlib.pyplot as plt

        # Plot and save 'timestamp'
        plt.figure(figsize=(12, 6))
        plt.plot(self.eet_df['timestamp'], label='Timestamp in ms')
        plt.xlabel('Sample')
        plt.ylabel('Time (seconds)')
        #plt.title('Raw Timestamp Plot')
        plt.legend()
        plt.grid()
        plt.savefig(f'timestamp_plot.png')
        plt.close()

        # Find discontinuities in timestamp
        time_diff = self.eet_df['time_seconds'].diff()
        discontinuities = self.eet_df[(time_diff > 1) | (time_diff < -1)]

        for i in discontinuities.index:
            offset = self.eet_df['time_seconds'].loc[i] - self.eet_df['time_seconds'].loc[i - 1]
            if offset > 1:
                self.eet_df.loc[i:, 'time_seconds'] -= offset
            else:
                self.eet_df.loc[i:, 'time_seconds'] += abs(offset)
                
        # Plot and save 'timestamp'
        plt.figure(figsize=(12, 6))
        plt.plot(self.eet_df['time_seconds'], label='Time in seconds')
        plt.xlabel('Sample')
        plt.ylabel('Time (seconds)')
        #plt.title('Time Plot')
        plt.legend()
        plt.grid()
        plt.savefig(f'time_plot.png')
        plt.close()

        # Remove irrelevant columns    
        self.eet_df = self.eet_df.drop(columns=['timestamp','combined_ray_origin', 'combined_ray_direction', 
                            'left_ray_origin', 'left_ray_direction',
                            'right_ray_origin', 'right_ray_direction'])

        # Remove uncalibrated data
        self.eet_df = self.eet_df[self.eet_df['calibration_valid'] == True]

        #self.eet_df = self.eet_df.reset_index(drop=True)

        # Remove data where left and right ray are unsynced
        self.eet_df = self.eet_df[(self.eet_df['left_ray_valid'] == self.eet_df['right_ray_valid'])]
        self.eet_df = self.eet_df[(self.eet_df['combined_ray_valid'] == self.eet_df['left_ray_valid'])]

        # Median filter
        from scipy.signal import medfilt
        binary_data = self.eet_df['combined_ray_valid'].astype(int).to_numpy()
        binary_data = medfilt(binary_data, kernel_size=5)

        # binary_data = 1 - binary_data
        # kernel_size = 5
        # kernel = np.ones(kernel_size) 
        # convolved = np.convolve(binary_data, kernel, mode='same')

        self.eet_df['filt'] = binary_data
        self.eet_df['filt'] = self.eet_df['filt'].astype(bool)



    def blinks_detection(self, df, threshold=0.1):
        blinks=[]
        for i in df.index:
            if df['filt'][i] == False:
                blink_start = df['time_seconds'][i]
                for j in df.index[i:]:
                    if df['filt'][j] == True:
                        blink_end = df['time_seconds'][j]
                        blink_duration = blink_end - blink_start
                        if (blink_duration >= threshold) & (blink_duration <= 0.3):
                            blinks.append((blink_start, blink_end))
                        break
                i=j
        return blinks

    def feature_extraction(self):
        for k in self.exc_df.index:
            spawn_time = self.exc_df['spawn_time'][k]
            destruction_time = self.exc_df['destruction_time'][k]
            mask = (self.eet_df['time_seconds'] >= spawn_time) & (self.eet_df['time_seconds'] <= destruction_time)
            instance = self.eet_df[mask]
            instance.reset_index(drop=True, inplace=True)
            blinks=self.blinks_detection(instance)
            BR = len(blinks) / (destruction_time - spawn_time) * 60  # blink rate per minute
            BD_mean = np.mean([blink[1]-blink[0] for blink in blinks]) if blinks else 0  # average blink duration
            BD_var = np.var([blink[1]-blink[0] for blink in blinks]) if blinks else 0  # variance of blink duration
            BD_std = np.std([blink[1]-blink[0] for blink in blinks]) if blinks else 0  # std of blink duration
            #print(k, len(blinks), BR, BD, len(instance))
            self.features["index"].append(k)
            self.features["blink_rate"].append(BR)
            self.features["blink_duration_mean"].append(BD_mean)
            self.features["blink_duration_var"].append(BD_var)
            self.features["blink_duration_std"].append(BD_std)
    
# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 6))
# plt.plot(self.eet_df.loc[0:500,'time_seconds'], self.eet_df.loc[0:500,'filt'], label='Convolved Signal', alpha=0.7)
# plt.plot(self.eet_df.loc[0:500,'time_seconds'], self.eet_df.loc[0:500,'combined_ray_valid'], label='Combined Ray Valid', alpha=0.7)
# plt.xlabel('Time (seconds)')
# plt.ylabel('Signal')
# plt.title('Convolved Signal vs Combined Ray Valid')
# plt.legend()
# plt.grid()
# plt.savefig(f'blink_detection.png')

# for i in range(100):
#     print(self.eet_df['time_seconds'][i], self.eet_df['combined_ray_valid'][i], self.eet_df['filt'][i])


        


    

