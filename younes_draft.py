import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils.visualizations import visualize

base_path = 'dataset/pediatric_echo_avi/A4C'

metadata = pd.read_csv(f'{base_path}/FileList.csv')
volume_tracings = pd.read_csv(f'{base_path}/VolumeTracings.csv')

print(metadata.describe())

visualize(metadata, volume_tracings, base_path)
#
# metadata = pd.get_dummies(metadata, columns=['Sex'], drop_first=True)
#
# metadata['EF_Binary'] = (metadata['EF'] < 50).astype(int)
#
# # Preprocess metadata (standardize demographic features)
# features = ['Age', 'Weight', 'Height']
# scaler = StandardScaler()
# metadata[features] = scaler.fit_transform(metadata[features])
#
# # Aggregate volume tracings for each video by calculating the mean X and Y
# volume_features = volume_tracings.groupby('FileName').agg({
#     'X': ['mean', 'std', 'min', 'max'],
#     'Y': ['mean', 'std', 'min', 'max']
# }).reset_index()
#
# # Flattening
# volume_features.columns = ['_'.join(col).strip() for col in volume_features.columns.values]
#
# # Merge volume features with metadata
# data = pd.merge(metadata, volume_features, left_on='FileName', right_on='FileName_', how='inner')
#
# data = data.dropna()
#
# # Drop irrelevant columns
# X = data.drop(['FileName', 'FileName_', 'EF', 'Split', 'EF_Binary'], axis=1)
# y = data['EF_Binary']
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# model = LogisticRegression(max_iter=1000)
# model.fit(X_train, y_train)
#
# y_pred = model.predict(X_test)
#
# accuracy = accuracy_score(y_test, y_pred)
# conf_matrix = confusion_matrix(y_test, y_pred)
# class_report = classification_report(y_test, y_pred)
#
# print(f'Accuracy: {accuracy:.2f}')
# print('Confusion Matrix:')
# print(conf_matrix)
# print('Classification Report:')
# print(class_report)
