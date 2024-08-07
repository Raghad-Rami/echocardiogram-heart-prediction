import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

base_path = 'dataset/pediatric_echo_avi/A4C'

metadata = pd.read_csv(f'{base_path}/FileList.csv')
volume_tracings = pd.read_csv(f'{base_path}/VolumeTracings.csv')

print(metadata.describe())

# sns.histplot(df['Age'], bins=20, kde=True)
# plt.title('Age Distribution')
# plt.show()

# filename = 'CR32a7555-CR32a7582-000039.avi'
# frame_number = 39
# tracings_for_frame = volume_tracings[(volume_tracings['FileName'] == filename) & (volume_tracings['Frame'] == str(frame_number))]
#
# cap = cv2.VideoCapture(f'{base_path}/Videos/{filename}')
# cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
# ret, frame = cap.read()
#
# plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
# plt.scatter(tracings_for_frame['X'], tracings_for_frame['Y'], color='red', s=5)
# plt.title(f'Frame {frame_number} Tracings')
# plt.show()
#
# cap.release()

metadata = pd.get_dummies(metadata, columns=['Sex'], drop_first=True)

metadata['EF_Binary'] = (metadata['EF'] < 50).astype(int)

# Preprocess metadata (standardize demographic features)
features = ['Age', 'Weight', 'Height']
scaler = StandardScaler()
metadata[features] = scaler.fit_transform(metadata[features])

# Aggregate volume tracings for each video by calculating the mean X and Y
volume_features = volume_tracings.groupby('FileName').agg({
    'X': ['mean', 'std', 'min', 'max'],
    'Y': ['mean', 'std', 'min', 'max']
}).reset_index()

# Flattening
volume_features.columns = ['_'.join(col).strip() for col in volume_features.columns.values]

# Merge volume features with metadata
data = pd.merge(metadata, volume_features, left_on='FileName', right_on='FileName_', how='inner')

data = data.dropna()

# Drop irrelevant columns
X = data.drop(['FileName', 'FileName_', 'EF', 'Split', 'EF_Binary'], axis=1)
y = data['EF_Binary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)
