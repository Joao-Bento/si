import pandas as pd
import numpy as np
import preprocessing

class run:
    def __init__(self, path):

        self.path = path

        si_path = f'{self.path}/spatial_input_data.csv'

        self.si_df = pd.read_csv(si_path)
        exercises = preprocessing.merge(self.path, 'spatial_input_data.csv')
        self.merger = exercises.merger
        
        self.process_data()
        self.features = {
            "id": [],
            "roll_mean_head": [],
            "roll_var_head": [],
            "roll_std_head": [],
            "roll_max_head": [],
            "roll_min_head": [],
            "yaw_mean_head": [],
            "yaw_var_head": [],
            "yaw_std_head": [],
            "yaw_max_head": [],
            "yaw_min_head": [],
            "pitch_mean_head": [],
            "pitch_var_head": [],
            "pitch_std_head": [],
            "pitch_max_head": [],
            "pitch_min_head": [],
            "linv_mean_head": [],
            "linv_var_head": [],
            "linv_std_head": [],
            "linv_max_head": [],
            "linv_min_head": [],
            "rollv_mean_head": [],
            "rollv_var_head": [],
            "rollv_std_head": [],
            "rollv_max_head": [],
            "rollv_min_head": [],
            "yawv_mean_head": [],
            "yawv_var_head": [],
            "yawv_std_head": [],
            "yawv_max_head": [],
            "yawv_min_head": [],
            "pitchv_mean_head": [],
            "pitchv_var_head": [],
            "pitchv_std_head": [],
            "pitchv_max_head": [],
            "pitchv_min_head": [],
            "roll_mean_lwrist": [],
            "roll_var_lwrist": [],
            "roll_std_lwrist": [],
            "roll_max_lwrist": [],
            "roll_min_lwrist": [],
            "yaw_mean_lwrist": [],
            "yaw_var_lwrist": [],
            "yaw_std_lwrist": [],
            "yaw_max_lwrist": [],
            "yaw_min_lwrist": [],
            "pitch_mean_lwrist": [],
            "pitch_var_lwrist": [],
            "pitch_std_lwrist": [],
            "pitch_max_lwrist": [],
            "pitch_min_lwrist": [],
            "linv_mean_lwrist": [],
            "linv_var_lwrist": [],
            "linv_std_lwrist": [],
            "linv_max_lwrist": [],
            "linv_min_lwrist": [],
            "roll_mean_rwrist": [],
            "roll_var_rwrist": [],
            "roll_std_rwrist": [],
            "roll_max_rwrist": [],
            "roll_min_rwrist": [],
            "yaw_mean_rwrist": [],
            "yaw_var_rwrist": [],
            "yaw_std_rwrist": [],
            "yaw_max_rwrist": [],
            "yaw_min_rwrist": [],
            "pitch_mean_rwrist": [],
            "pitch_var_rwrist": [],
            "pitch_std_rwrist": [],
            "pitch_max_rwrist": [],
            "pitch_min_rwrist": [],
            "linv_mean_rwrist": [],
            "linv_var_rwrist": [],
            "linv_std_rwrist": [],
            "linv_max_rwrist": [],
            "linv_min_rwrist": []
        }
        
        self.feature_extraction()
        
    def process_data(self):
        ti_si = self.si_df['timestamp'][0]
        self.si_df['time_seconds'] = (self.si_df['timestamp'] - ti_si) / 10**7

        # Find discontinuities in timestamp
        # time_diff = self.si_df['time_seconds'].diff()
        # discontinuities = self.si_df[(time_diff > 1) | (time_diff < -1)]

        # for i in discontinuities.index:
        #     offset = self.si_df['time_seconds'].loc[i] - self.si_df['time_seconds'].loc[i - 1]
        #     if offset > 1:
        #         self.si_df.loc[i:, 'time_seconds'] -= offset
        #     else:
        #         self.si_df.loc[i:, 'time_seconds'] += abs(offset)
                
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
        self.hp_df['valid'] = self.hp_df['head_pose_valid']

        self.lwp_df = self.si_df.drop(columns=['timestamp', 'head_pose_valid', 'head_position', 'head_forward',
            'head_up', 'eye_ray_valid', 'eye_ray_origin', 'eye_ray_direction','hand_right_valid',
            'right_wrist_position', 'right_wrist_orientation', 'right_wrist_radius',
            'right_wrist_accuracy'])
        self.lwp_df['valid'] = self.lwp_df['hand_left_valid']
        
        self.rwp_df = self.si_df.drop(columns=['timestamp', 'head_pose_valid', 'head_position', 'head_forward',
            'head_up', 'eye_ray_valid', 'eye_ray_origin', 'eye_ray_direction', 'hand_left_valid',
            'left_wrist_position', 'left_wrist_orientation', 'left_wrist_radius',
            'left_wrist_accuracy'])
        self.rwp_df['valid'] = self.rwp_df['hand_right_valid']

        # Find indexes where 'head_forward' is zero
        #zero_head_forward_indexes = self.hp_df[self.hp_df['head_forward'] == [0 0 0]].index
        #print("Indexes where 'head_forward' is zero:", zero_head_forward_indexes)


        # Remove uncalibrated data
        self.hp_df = self.hp_df[self.hp_df['head_pose_valid'] == True]
        self.lwp_df = self.lwp_df[self.lwp_df['hand_left_valid'] == True]
        self.rwp_df = self.rwp_df[self.rwp_df['hand_right_valid'] == True]

        # Transform data from string to numpy array
        self.hp_df['head_position'] = self.hp_df['head_position'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.hp_df['head_forward'] = self.hp_df['head_forward'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.hp_df['head_up'] = self.hp_df['head_up'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        
        self.lwp_df['left_wrist_position'] = self.lwp_df['left_wrist_position'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.lwp_df['left_wrist_orientation'] = self.lwp_df['left_wrist_orientation'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.rwp_df['right_wrist_position'] = self.rwp_df['right_wrist_position'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
        self.rwp_df['right_wrist_orientation'] = self.rwp_df['right_wrist_orientation'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))

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
        
        if not df['valid'].any():
            return [0] * 20
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
        
        if not df['valid'].any():
            return [[0]*5, [0]*5, [0]*5]
        
        return ori_features

    def feature_extraction(self):
        for id in self.merger[:,0]:
            id = int(id)
            spawn_index = int(self.merger[id-1][1])
            destruction_index = int(self.merger[id-1][2])
            #mask = (self.eet_df['time_seconds'] >= spawn_time) & (self.eet_df['time_seconds'] <= destruction_time)
            instance_hp = self.hp_df.loc[spawn_index:destruction_index+1]
            instance_lwp = self.lwp_df.loc[spawn_index:destruction_index+1]
            instance_rwp = self.rwp_df.loc[spawn_index:destruction_index+1]
            # if len(instance_hp) < 45: #frequency/2:
            #     print('broke')
            #     continue
            roll_features_head = self.orientation_features(instance_hp, 'head_forward')
            yaw_features_head = self.orientation_features(instance_hp, 'head_up')
            pitch_features_head = self.orientation_features(instance_hp, 'head_left')
            linv_features_head = self.velocity_features(instance_hp, 'head_position')
            rollv_features_head = self.velocity_features(instance_hp, 'head_forward')
            yawv_features_head = self.velocity_features(instance_hp, 'head_up')
            pitchv_features_head = self.velocity_features(instance_hp, 'head_left')
            ori_features_lwrist = self.orientation_features(instance_lwp, 'left_wrist_orientation')
            linv_features_lwrist = self.velocity_features(instance_lwp, 'left_wrist_position')
            ori_features_rwrist = self.orientation_features(instance_rwp, 'right_wrist_orientation')
            linv_features_rwrist = self.velocity_features(instance_rwp, 'right_wrist_position')
            self.features["id"].append(id)
            self.features["roll_mean_head"].append(roll_features_head[0][0])
            self.features["roll_var_head"].append(roll_features_head[0][1])
            self.features["roll_std_head"].append(roll_features_head[0][2])
            self.features["roll_max_head"].append(roll_features_head[0][3])
            self.features["roll_min_head"].append(roll_features_head[0][4])
            self.features["yaw_mean_head"].append(yaw_features_head[1][0])
            self.features["yaw_var_head"].append(yaw_features_head[1][1])
            self.features["yaw_std_head"].append(yaw_features_head[1][2])
            self.features["yaw_max_head"].append(yaw_features_head[1][3])
            self.features["yaw_min_head"].append(yaw_features_head[1][4])
            self.features["pitch_mean_head"].append(pitch_features_head[2][0])
            self.features["pitch_var_head"].append(pitch_features_head[2][1])
            self.features["pitch_std_head"].append(pitch_features_head[2][2])
            self.features["pitch_max_head"].append(pitch_features_head[2][3])
            self.features["pitch_min_head"].append(pitch_features_head[2][4])
            self.features["linv_mean_head"].append(linv_features_head[0])
            self.features["linv_var_head"].append(linv_features_head[1])
            self.features["linv_std_head"].append(linv_features_head[2])
            self.features["linv_max_head"].append(linv_features_head[3])
            self.features["linv_min_head"].append(linv_features_head[4])
            self.features["rollv_mean_head"].append(rollv_features_head[0])
            self.features["rollv_var_head"].append(rollv_features_head[1])
            self.features["rollv_std_head"].append(rollv_features_head[2])
            self.features["rollv_max_head"].append(rollv_features_head[3])
            self.features["rollv_min_head"].append(rollv_features_head[4])
            self.features["yawv_mean_head"].append(yawv_features_head[0])
            self.features["yawv_var_head"].append(yawv_features_head[1])
            self.features["yawv_std_head"].append(yawv_features_head[2])
            self.features["yawv_max_head"].append(yawv_features_head[3])
            self.features["yawv_min_head"].append(yawv_features_head[4])
            self.features["pitchv_mean_head"].append(pitchv_features_head[0])
            self.features["pitchv_var_head"].append(pitchv_features_head[1])
            self.features["pitchv_std_head"].append(pitchv_features_head[2])
            self.features["pitchv_max_head"].append(pitchv_features_head[3])
            self.features["pitchv_min_head"].append(pitchv_features_head[4])
            self.features["roll_mean_lwrist"].append(ori_features_lwrist[0][0])
            self.features["roll_var_lwrist"].append(ori_features_lwrist[0][1])
            self.features["roll_std_lwrist"].append(ori_features_lwrist[0][2])
            self.features["roll_max_lwrist"].append(ori_features_lwrist[0][3])
            self.features["roll_min_lwrist"].append(ori_features_lwrist[0][4])
            self.features["yaw_mean_lwrist"].append(ori_features_lwrist[1][0])
            self.features["yaw_var_lwrist"].append(ori_features_lwrist[1][1])
            self.features["yaw_std_lwrist"].append(ori_features_lwrist[1][2])
            self.features["yaw_max_lwrist"].append(ori_features_lwrist[1][3])
            self.features["yaw_min_lwrist"].append(ori_features_lwrist[1][4])
            self.features["pitch_mean_lwrist"].append(ori_features_lwrist[2][0])
            self.features["pitch_var_lwrist"].append(ori_features_lwrist[2][1])
            self.features["pitch_std_lwrist"].append(ori_features_lwrist[2][2])
            self.features["pitch_max_lwrist"].append(ori_features_lwrist[2][3])
            self.features["pitch_min_lwrist"].append(ori_features_lwrist[2][4])
            self.features["linv_mean_lwrist"].append(linv_features_lwrist[0])
            self.features["linv_var_lwrist"].append(linv_features_lwrist[1])
            self.features["linv_std_lwrist"].append(linv_features_lwrist[2])
            self.features["linv_max_lwrist"].append(linv_features_lwrist[3])
            self.features["linv_min_lwrist"].append(linv_features_lwrist[4])
            self.features["roll_mean_rwrist"].append(ori_features_rwrist[0][1])
            self.features["roll_var_rwrist"].append(ori_features_rwrist[0][1])
            self.features["roll_std_rwrist"].append(ori_features_rwrist[0][2])
            self.features["roll_max_rwrist"].append(ori_features_rwrist[0][3])
            self.features["roll_min_rwrist"].append(ori_features_rwrist[0][4])
            self.features["yaw_mean_rwrist"].append(ori_features_rwrist[1][0])
            self.features["yaw_var_rwrist"].append(ori_features_rwrist[1][1])
            self.features["yaw_std_rwrist"].append(ori_features_rwrist[1][2])
            self.features["yaw_max_rwrist"].append(ori_features_rwrist[1][3])
            self.features["yaw_min_rwrist"].append(ori_features_rwrist[1][4])
            self.features["pitch_mean_rwrist"].append(ori_features_rwrist[2][0])
            self.features["pitch_var_rwrist"].append(ori_features_rwrist[2][1])
            self.features["pitch_std_rwrist"].append(ori_features_rwrist[2][2])
            self.features["pitch_max_rwrist"].append(ori_features_rwrist[2][3])
            self.features["pitch_min_rwrist"].append(ori_features_rwrist[2][4])
            self.features["linv_mean_rwrist"].append(linv_features_rwrist[0])
            self.features["linv_var_rwrist"].append(linv_features_rwrist[1])
            self.features["linv_std_rwrist"].append(linv_features_rwrist[2])
            self.features["linv_max_rwrist"].append(linv_features_rwrist[3])
            self.features["linv_min_rwrist"].append(linv_features_rwrist[4])

            
    
    