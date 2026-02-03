import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class eye_movement_detection:
    def __init__(self, df, frequency, fov_x, fov_y, v_threshold):
        self.df = df
        self.frequency = frequency
        self.fov_x = fov_x
        self.fov_y = fov_y
        self.v_threshold = 20# v_threshold
        self.FF = -1
        self.AFD = -1
        self.SF = -1
        self.PSV = -1
        self.ASV = -1
        self.ASD = -1
        
        self.observations = None
        
        
        self.normalization()
        self.ivt()


    # df['time_seconds'] = (df['timestamp'] - df['timestamp'][0]) / 10**7
    # print(df['time_seconds'][frequency])
    # print("Total time (minutes):")
    # print(df['time_seconds'][len(df)-1]/60)


    def normalization(self):
        def string2array(s):
            s = s.strip('[]')
            return [float(num) for num in s.split()]

        self.df['combined_ray_direction'] = self.df['combined_ray_direction'].apply(string2array)
        self.df['x_combined'] = np.atan2(self.df['combined_ray_direction'].apply(lambda x: x[0]), self.df['combined_ray_direction'].apply(lambda x: x[2]))
        self.df['y_combined'] = np.atan2(self.df['combined_ray_direction'].apply(lambda x: x[1]), self.df['combined_ray_direction'].apply(lambda x: x[2]))

        self.df['x_combined'] = np.degrees(self.df['x_combined'])
        self.df['y_combined'] = np.degrees(self.df['y_combined'])

        # Normalize x_combined and y_combined according to the field of view (FoV)
        #self.df['x_combined'] = (self.df['x_combined']+self.fov_x) / (2*self.fov_x)
        #self.df['y_combined'] = (self.df['y_combined']+self.fov_y) / (2*self.fov_y)

    def ivt(self):
    #Calculate point-to-point velocities
        self.df['velocity'] = np.sqrt(
            (self.df['x_combined'].diff() ** 2) + (self.df['y_combined'].diff() ** 2)
        ) / self.df['time_seconds'].diff()


        self.df = self.df[self.df['combined_ray_valid'] == True]

        # max_velocity = self.df['velocity'].max()
        # min_velocity = self.df['velocity'].min()
        # print(f"Maximum velocity: {max_velocity}")
        # print(f"Minimum velocity: {min_velocity}")
    # print(f"Max x value: {df['x_combined'].max()}")
    # print(f"Min x value: {df['x_combined'].min()}")
    # print(f"Max y value: {df['y_combined'].max()}")
    # print(f"Min y value: {df['y_combined'].min()}")

        self.df['fixations'] = self.df['velocity'] < self.v_threshold
    # print(f"Percentage of fixations: {(df['fixations'].sum()/len(df))*100:.2f}%")
        self.observations = self.df['fixations']

        irwin = 0.15
        #bahill = (0.03)
        self.df['prediction'] = 0
        fixations = []
        saccades = []
        f=0
        while f < len(self.df):
            if self.df['fixations'].iloc[f]:
                fixation_start = f#self.df['ef_time'].iloc[f]
                while self.df['fixations'].iloc[f] and f < len(self.df)-1:
                    f += 1
                fixation_end = f-1#self.df['ef_time'].iloc[f]
                if self.df['time_seconds'].iloc[fixation_end] - self.df['time_seconds'].iloc[fixation_start] >= irwin: #Irwin 1992
                    fixations.append((fixation_start, fixation_end))
                    self.df.loc[fixation_start:fixation_end, 'prediction'] = 1
                else:
                    saccades.append((fixation_start, fixation_end))
                    self.df.loc[fixation_start:fixation_end, 'prediction'] = 2
                #print("Detected fixations", (fixation_start, fixation_end))
            else:
                saccade_start = f
                while not self.df['fixations'].iloc[f] and f < len(self.df)-1:
                    f += 1
                saccade_end = f-1
                if self.df['time_seconds'].iloc[saccade_end] - self.df['time_seconds'].iloc[saccade_start] < irwin: #Irwin 1992
                    saccades.append((saccade_start, saccade_end))
                    self.df.loc[saccade_start:saccade_end, 'prediction'] = 2
            f += 1
            
        self.FF = len(fixations)/(self.df['time_seconds'].iloc[-1]-self.df['time_seconds'].iloc[0])*60
        
        fixation_time = [(self.df['time_seconds'].iloc[start], self.df['time_seconds'].iloc[end]) for start, end in fixations]
        self.AFD = np.mean([end - start for start, end in fixation_time])
        
        self.SF = len(saccades)/(self.df['time_seconds'].iloc[-1]-self.df['time_seconds'].iloc[0])*60
        
        saccade_time = [(self.df['time_seconds'].iloc[start], self.df['time_seconds'].iloc[end]) for start, end in saccades]
        self.ASD = np.mean([end - start for start, end in saccade_time])
        
        self.df['velocity'] = self.df['velocity'].fillna(0)
        saccade_velocities = [(self.df['velocity'].iloc[start:end+1].max()) for start, end in saccades if end+1 > start]
        if saccade_velocities!=[]:
            self.ASV = np.mean(saccade_velocities)
            self.PSV = np.max(saccade_velocities)
        else:
            self.ASV = 0
            self.PSV = 0
        


# print(f'Experimental number of fixations: {len(fixations)}')
# #print(f'Ground truth number of fixations: {len(df_gt)}')
# print(f'Experimental average fixation duration: {avg_fixation_duration:.3f} seconds')
# #print(f'Ground truth average fixation duration: {avg_fixation_duration_gt:.3f} seconds')
# print(f"Percentage of fixations: {(df['prediction'].sum()/len(df))*100:.2f}%")

