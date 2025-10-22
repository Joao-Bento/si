import pandas as pd
import numpy as np

class run:
    def __init__(self, path):

        self.path = path

        exc_path = f'{self.path}/spheres.csv'
        si_path = f'{self.path}/spatial_input_data.csv'

        self.exc_df = pd.read_csv(exc_path)
        self.si_df = pd.read_csv(si_path)
        
        self.process_data()
        self.features = {
            "index": [],
            "roll_mean": [],
            "roll_var": [],
            "roll_std": [],
            "roll_max": [],
            "roll_min": [],
            "yaw_mean": [],
            "yaw_var": [],
            "yaw_std": [],
            "yaw_max": [],
            "yaw_min": [],
            "pitch_mean": [],
            "pitch_var": [],
            "pitch_std": [],
            "pitch_max": [],
            "pitch_min": [],
            "linv_mean": [],
            "linv_var": [],
            "linv_std": [],
            "linv_max": [],
            "linv_min": [],
            "rollv_mean": [],
            "rollv_var": [],
            "rollv_std": [],
            "rollv_max": [],
            "rollv_min": [],
            "yawv_mean": [],
            "yawv_var": [],
            "yawv_std": [],
            "yawv_max": [],
            "yawv_min": [],
            "pitchv_mean": [],
            "pitchv_var": [],
            "pitchv_std": [],
            "pitchv_max": [],
            "pitchv_min": []
        } 
        
        self.feature_extraction()
        
    def process_data(self):
        ti_spawn_exc = self.exc_df['spawn_time'][0]
        ti_destruction_exc = self.exc_df['destruction_time'][0]
        ti_si = self.si_df['timestamp'][0]
        self.exc_df['spawn_time'] = self.exc_df['spawn_time'] - ti_spawn_exc
        self.exc_df['destruction_time'] = self.exc_df['destruction_time'] - ti_spawn_exc
        self.si_df['time_seconds'] = (self.si_df['timestamp'] - ti_si) / 10**7

        # Find discontinuities in timestamp
        time_diff = self.si_df['time_seconds'].diff()
        discontinuities = self.si_df[(time_diff > 1) | (time_diff < -1)]

        for i in discontinuities.index:
            offset = self.si_df['time_seconds'].loc[i] - self.si_df['time_seconds'].loc[i - 1]
            if offset > 1:
                self.si_df.loc[i:, 'time_seconds'] -= offset
            else:
                self.si_df.loc[i:, 'time_seconds'] += abs(offset)
                
        # import matplotlib.pyplot as plt

        # plt.plot(self.si_df['time_seconds'])
        # plt.xlabel('Index')
        # plt.ylabel('Time (seconds)')
        # plt.title('Time Seconds Plot')
        # plt.savefig('time_seconds_plot.png')

        # Separate head and wrist data  
        self.hp_df = self.si_df.drop(columns=['timestamp','eye_ray_valid', 'eye_ray_origin', 'eye_ray_direction',
            'hand_left_valid', 'left_wrist_position', 'left_wrist_orientation',
            'left_wrist_radius', 'left_wrist_accuracy', 'hand_right_valid',
            'right_wrist_position', 'right_wrist_orientation', 'right_wrist_radius',
            'right_wrist_accuracy'])

        self.wp_df = self.si_df.drop(columns=['timestamp', 'head_pose_valid', 'head_position', 'head_forward',
            'head_up', 'eye_ray_valid', 'eye_ray_origin', 'eye_ray_direction'])

        # Find indexes where 'head_forward' is zero
        #zero_head_forward_indexes = self.hp_df[self.hp_df['head_forward'] == [0 0 0]].index
        #print("Indexes where 'head_forward' is zero:", zero_head_forward_indexes)


        # Remove uncalibrated data
        self.hp_df = self.hp_df[self.hp_df['head_pose_valid'] == True]

        # Transform data from string to numpy array
        self.hp_df['head_position'] = self.hp_df['head_position'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.hp_df['head_forward'] = self.hp_df['head_forward'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.hp_df['head_up'] = self.hp_df['head_up'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

        # Calculate the  component (head_left) using the cross product of head_forward and head_up
        self.hp_df['head_left'] = self.hp_df.apply(lambda row: np.cross(row['head_forward'], row['head_up']), axis=1)

        
    def velocity_features(self,df,col):
        time_diff = df['time_seconds'].diff()
        r_diff = df[col].diff().apply(lambda x: np.linalg.norm(x))
        vel_r = r_diff / time_diff
        x_diff = df[col].diff().apply(lambda x: x[0] if isinstance(x, np.ndarray) else np.nan)  
        vel_x = x_diff / time_diff  
        y_diff = df[col].diff().apply(lambda x: x[1] if isinstance(x, np.ndarray) else np.nan)
        vel_y = y_diff / time_diff
        z_diff = df[col].diff().apply(lambda x: x[2] if isinstance(x, np.ndarray) else np.nan)
        vel_z = z_diff / time_diff
        
        vel_r_mean = vel_r.mean()
        vel_r_var = vel_r.var()
        vel_r_std = vel_r.std()
        vel_r_max = vel_r.max()
        vel_r_min = vel_r.min()

        vel_x_mean = vel_x.mean()
        vel_x_var = vel_x.var()
        vel_x_std = vel_x.std()
        vel_x_max = vel_x.max()
        vel_x_min = vel_x.min()

        vel_y_mean = vel_y.mean()
        vel_y_var = vel_y.var()
        vel_y_std = vel_y.std()
        vel_y_max = vel_y.max()
        vel_y_min = vel_y.min()

        vel_z_mean = vel_z.mean()
        vel_z_var = vel_z.var()
        vel_z_std = vel_z.std()
        vel_z_max = vel_z.max()
        vel_z_min = vel_z.min()
        
        vel_features = [vel_r_mean, vel_r_var, vel_r_std, vel_r_max, vel_r_min,
                        vel_x_mean, vel_x_var, vel_x_std, vel_x_max, vel_x_min,
                        vel_y_mean, vel_y_var, vel_y_std, vel_y_max, vel_y_min,
                        vel_z_mean, vel_z_var, vel_z_std, vel_z_max, vel_z_min]
        return vel_features

    def orientation_features(self,df,col):
        x = df[col].apply(lambda v: v[0] if isinstance(v, np.ndarray) else np.nan)
        y = df[col].apply(lambda v: v[1] if isinstance(v, np.ndarray) else np.nan)
        z = df[col].apply(lambda v: v[2] if isinstance(v, np.ndarray) else np.nan)
        
        def compute_stats(component):
            mean = component.mean()
            var = component.var()
            std = component.std()
            max_val = component.max()
            min_val = component.min()
            return [mean, var, std, max_val, min_val]
        
        x_stats = compute_stats(x)
        y_stats = compute_stats(y)
        z_stats = compute_stats(z)
        ori_features = [x_stats, y_stats, z_stats]
        
        return ori_features

    def feature_extraction(self):
        for k in self.exc_df.index:
            spawn_time = self.exc_df['spawn_time'][k]
            destruction_time = self.exc_df['destruction_time'][k]
            mask = (self.hp_df['time_seconds'] >= spawn_time) & (self.hp_df['time_seconds'] <= destruction_time)
            instance = self.hp_df[mask]
            roll_features = self.orientation_features(instance, 'head_forward')
            yaw_features = self.orientation_features(instance, 'head_up')
            pitch_features = self.orientation_features(instance, 'head_left')
            linv_features = self.velocity_features(instance, 'head_position')
            rollv_features = self.velocity_features(instance, 'head_forward')
            yawv_features = self.velocity_features(instance, 'head_up')
            pitchv_features = self.velocity_features(instance, 'head_left')
            self.features["index"].append(k)
            self.features["roll_mean"].append(roll_features[0][0])
            self.features["roll_var"].append(roll_features[0][1])
            self.features["roll_std"].append(roll_features[0][2])
            self.features["roll_max"].append(roll_features[0][3])
            self.features["roll_min"].append(roll_features[0][4])
            self.features["yaw_mean"].append(yaw_features[1][0])
            self.features["yaw_var"].append(yaw_features[1][1])
            self.features["yaw_std"].append(yaw_features[1][2])
            self.features["yaw_max"].append(yaw_features[1][3])
            self.features["yaw_min"].append(yaw_features[1][4])
            self.features["pitch_mean"].append(pitch_features[2][0])
            self.features["pitch_var"].append(pitch_features[2][1])
            self.features["pitch_std"].append(pitch_features[2][2])
            self.features["pitch_max"].append(pitch_features[2][3])
            self.features["pitch_min"].append(pitch_features[2][4])
            self.features["linv_mean"].append(linv_features[0])
            self.features["linv_var"].append(linv_features[1])
            self.features["linv_std"].append(linv_features[2])
            self.features["linv_max"].append(linv_features[3])
            self.features["linv_min"].append(linv_features[4])
            self.features["rollv_mean"].append(rollv_features[0])
            self.features["rollv_var"].append(rollv_features[1])
            self.features["rollv_std"].append(rollv_features[2])
            self.features["rollv_max"].append(rollv_features[3])
            self.features["rollv_min"].append(rollv_features[4])
            self.features["yawv_mean"].append(yawv_features[0])
            self.features["yawv_var"].append(yawv_features[1])
            self.features["yawv_std"].append(yawv_features[2])
            self.features["yawv_max"].append(yawv_features[3])
            self.features["yawv_min"].append(yawv_features[4])
            self.features["pitchv_mean"].append(pitchv_features[0])
            self.features["pitchv_var"].append(pitchv_features[1])
            self.features["pitchv_std"].append(pitchv_features[2])
            self.features["pitchv_max"].append(pitchv_features[3])
            self.features["pitchv_min"].append(pitchv_features[4])
    
    
    