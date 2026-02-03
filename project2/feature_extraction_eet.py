import pandas as pd
import numpy as np
import ivt, ihmm
import preprocessing

# runs by exercise
class run:
    def __init__(self, path):

        self.path = path

        eet_path = f'{self.path}/eet_data.csv'

        self.eet_df = pd.read_csv(eet_path)
        exercises = preprocessing.merge(self.path, 'eet_data.csv')
        self.merger = exercises.merger
        
        self.process_data()
        self.features = {
            "id": [],
            "blink_rate": [],
            "blink_duration_mean": [],
            "blink_duration_var": [],
            "blink_duration_std": [],
            "fixation_frequency": [],
            "fixation_duration_mean": [],
            "saccade_frequency": [],
            "saccade_duration_mean": [],
            "saccade_velocity_mean": [],
            "peak_saccade_velocity": []}
        self.feature_extraction()
        
    def process_data(self):
        
        ti_eet = self.eet_df['timestamp'][0]
        self.eet_df['time_seconds'] = (self.eet_df['timestamp'] - ti_eet) / 10**7
        
        # Remove uncalibrated and pre-acquisition data
        self.eet_df = self.eet_df[self.eet_df['calibration_valid'] == True]
        self.eet_df = self.eet_df[self.eet_df['id'] != -1]

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
        #true_count = self.eet_df['filt'].sum()
        #print(f"Number of entries where 'filt' is True: {true_count}")



    def blinks_detection(self, df, threshold=0.1):
        blinks=[]
        self.eet_df['blink'] = False
        for i in df.index:
            if df['filt'].loc[i] == False:
                blink_start = df['time_seconds'].loc[i]
                if len(df.loc[i:]) == 0:
                    break
                for j in df.loc[i:].index:
                    if df['filt'][j] == True:
                        blink_end = df['time_seconds'][j]
                        blink_duration = blink_end - blink_start
                        if (blink_duration >= threshold) & (blink_duration <= 0.3):
                            blinks.append((blink_start, blink_end))
                            self.eet_df.loc[i:j, 'blink'] = True
                        break
                i=j
        return blinks
    
    def fixation_detection(self, df):
        sampling_frequency = 90
        fov_x = 43
        fov_y = 29
        v_threshold = 20
        emd = ihmm.eye_movement_detection(df, sampling_frequency, fov_x, fov_y, v_threshold)
        FF = emd.FF
        AFD = emd.AFD
        SF = emd.SF
        ASD = emd.ASD
        ASV = emd.ASV
        PSV = emd.PSV
        self.eet_df['fixation'] = emd.observations
        return FF, AFD, SF, ASD, ASV, PSV
        
        
    def feature_extraction(self):
        #print(f'Number of exercises to process: {self.merger}')
        for id in self.merger[:,0]:
            id = int(id)
            spawn_index = int(self.merger[id-1][1])
            destruction_index = int(self.merger[id-1][2])
            #mask = (self.eet_df['time_seconds'] >= spawn_time) & (self.eet_df['time_seconds'] <= destruction_time)
            instance = self.eet_df.loc[spawn_index:destruction_index+1]
            # if len(instance) < 45: #frequency/2:
            #     print('broke')
            #     continue
            #instance.reset_index(drop=True, inplace=True)
            blinks=self.blinks_detection(instance)
            BR = len(blinks) / (instance['time_seconds'].iloc[-1] - instance['time_seconds'].iloc[0]) *60   # blink rate per minute
            BD_mean = np.mean([blink[1]-blink[0] for blink in blinks]) if blinks else 0  # average blink duration
            BD_var = np.var([blink[1]-blink[0] for blink in blinks]) if blinks else 0  # variance of blink duration
            BD_std = np.std([blink[1]-blink[0] for blink in blinks]) if blinks else 0  # std of blink duration
            #self.eet_df = self.eet_df[self.eet_df['calibration_valid'] == True]
            #instance = instance[instance['calibration_valid'] == True]
            FF, AFD, SF, ASD, ASV, PSV = self.fixation_detection(instance)
            #print(k, len(blinks), BR, BD, len(instance))
            self.features["id"].append(id)
            self.features["blink_rate"].append(BR)
            self.features["blink_duration_mean"].append(BD_mean)
            self.features["blink_duration_var"].append(BD_var)
            self.features["blink_duration_std"].append(BD_std)
            self.features["fixation_frequency"].append(FF)
            self.features["fixation_duration_mean"].append(AFD)
            self.features["saccade_frequency"].append(SF)
            self.features["saccade_duration_mean"].append(ASD)
            self.features["saccade_velocity_mean"].append(ASV)
            self.features["peak_saccade_velocity"].append(PSV)
            
# if __name__ == "__main__":
#     path = "/home/joao/tese/data_acquisition/dataset/010"
#     feature_extractor = run(path)
#     print(feature_extractor.features)
    # # Check if 'blink' and 'fixation' match in any row
    # matching_rows = feature_extractor.eet_df[
    #     (feature_extractor.eet_df['blink'] == True) & 
    #     (feature_extractor.eet_df['fixation'] == False)
    # ]
    # if not matching_rows.empty:
    #     print("There are rows where 'blink' and 'fixation' both are True.")
    #     print(len(matching_rows), len(feature_extractor.eet_df))
    # else:
    #     print("No rows found where 'blink' and 'fixation' both are True.")