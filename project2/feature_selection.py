import feature_extraction_eet, feature_extraction_si, feature_extraction_imu
import pandas as pd
import seaborn as sns
import os

# runs by individual
dataset_path = '/home/joao/tese/data_acquisition/dataset'
output_path = '/home/joao/si/project2/all_features.csv'


folders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
folders = [f for f in folders if f not in ['007', '011.2', '002']]
#folders = ['012']

df_features = pd.DataFrame()
for folder in folders:
    print(f"Processing folder: {folder}")
    path = os.path.join(dataset_path, folder)
    def extract_all_features(path):
        blinks = feature_extraction_eet.run(path)
        head_pose = feature_extraction_si.run(path)
        accelerometer = feature_extraction_imu.run(path, 'acc')
        gyroscope = feature_extraction_imu.run(path, 'gyro')
        magnetometer = feature_extraction_imu.run(path, 'mag')

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

    all_features= extract_all_features(path)
    features_names = list(all_features.keys())
    df = pd.DataFrame(columns=features_names)
    # for feature_list in all_features:
    #     for i in range(1,len(all_features)):
    #         row = {feature: feature_list[feature][i] for feature in features_names}
    #         df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    for feature in all_features:
        print(f"Feature: {feature}, Length: {len(all_features[feature])}")
    for i in range(len(all_features['id'])):
        row = {feature: all_features[feature][i] for feature in features_names}
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
            
    df['label'] = df['id'].apply(lambda x: 0 if 1 <= x <= 10 else (1 if 11 <= x <= 20 else None))
    df.drop(columns=['id'], inplace=True)
    df_features = pd.concat([df_features, df], ignore_index=True)
    
df_features.to_csv(output_path, index=False)

correlation_matrix = df_features.corr()
#print(correlation_matrix)
cm_cols = correlation_matrix.columns
cm_cols = cm_cols.drop('label')
# Drop features with high correlation
threshold = 0.8
to_drop = set()
for i in cm_cols:
    for j in cm_cols:
        if i != j and abs(correlation_matrix.loc[i, j]) > threshold:
            to_drop.add(j)
            
       
# Drop features with low correlation with the label
label_correlation = correlation_matrix['label']
low_correlation_features = label_correlation[abs(label_correlation) < 0.1].index
to_drop.update(low_correlation_features)

df_features = df_features.drop(columns=to_drop)
print(f"Number of features dropped: {len(to_drop)} out of {len(cm_cols)} total features; Dropped features: {to_drop}")

output_path = '/home/joao/si/project2/selected_features.csv'
df_features.to_csv(output_path, index=False)

correlation_matrix_selected = df_features.corr()


# # Optionally, save the correlation matrix to a CSV file
# #correlation_matrix.to_csv('correlation_matrix.csv')

# import matplotlib.pyplot as plt

# # Plot the correlation matrix
# plt.figure(figsize=(10, 10))
# sns.heatmap(correlation_matrix_selected, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)

# # Save the plot to a file
# plt.title("Correlation Matrix of Selected Features")
# plt.savefig("correlation_matrix_selected.png")
# plt.close()


#melhorar feature selection
#fazer leave 1 out cross validation
#testar outros modelos alem da nn
#fazer um fuzzy set
#