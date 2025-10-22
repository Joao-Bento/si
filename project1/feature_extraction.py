import preprocessing_eet, preprocessing_si, preprocessing_imu
import pandas as pd
import seaborn as sns

path_engaged = '/home/joao/tese/data_acquisition/engaged/001'
path_nonengaged = '/home/joao/tese/data_acquisition/nonengaged/001'

def extract_all_features(path):
    blinks = preprocessing_eet.run(path)
    head_pose = preprocessing_si.run(path)
    accelerometer = preprocessing_imu.run(path, 'acc')
    gyroscope = preprocessing_imu.run(path, 'gyro')
    magnetometer = preprocessing_imu.run(path, 'mag')

    blink_features = blinks.features
    head_pose_features = head_pose.features
    acc_features = accelerometer.features
    gyro_features = gyroscope.features
    mag_features = magnetometer.features

    all_features = {
        **blink_features,
        **head_pose_features,
        **acc_features,
        **gyro_features,
        **mag_features,
    }
    return all_features

all_features_engaged = extract_all_features(path_engaged)
all_features_nonengaged = extract_all_features(path_nonengaged)
all_features_engaged['label'] = [0] * len(next(iter(all_features_engaged.values())))
all_features_nonengaged['label'] = [1] * len(next(iter(all_features_nonengaged.values())))

features_list = [all_features_engaged, all_features_nonengaged]

output_path = 'all_features.csv'
features_names = list(all_features_engaged.keys())
df = pd.DataFrame(columns=features_names)
for feature_list in features_list:
    for i in feature_list['index']:
        row = {feature: feature_list[feature][i] for feature in features_names}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
df.drop(columns=['index'], inplace=True)
    
print(df.head())
df.to_csv(output_path, index=False)

correlation_matrix = df.corr()
#print(correlation_matrix)

# Drop features with high correlation
threshold = 0.8
to_drop = set()
for i in correlation_matrix.columns:
    for j in correlation_matrix.columns:
        if i != j and abs(correlation_matrix.loc[i, j]) > threshold:
            to_drop.add(j)
            
       
# Drop features with low correlation with the label
label_correlation = correlation_matrix['label']
low_correlation_features = label_correlation[abs(label_correlation) < 0.2].index
to_drop.update(low_correlation_features)

df = df.drop(columns=to_drop)
print(f"Number of features dropped: {len(to_drop)}; Dropped features: {to_drop}")

output_path = 'selected_features.csv'
df.to_csv(output_path, index=False)

correlation_matrix_selected = df.corr()


# Optionally, save the correlation matrix to a CSV file
#correlation_matrix.to_csv('correlation_matrix.csv')

import matplotlib.pyplot as plt

# Plot the correlation matrix
plt.figure(figsize=(10, 10))
sns.heatmap(correlation_matrix_selected, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# Save the plot to a file
plt.title("Correlation Matrix of Selected Features")
plt.savefig("correlation_matrix_selected.png")
plt.close()


#melhorar feature selection
#fazer leave 1 out cross validation
#testar outros modelos alem da nn
#fazer um fuzzy set
#
