from keras import Model
from keras.src.applications.vgg16 import VGG16
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import TimeDistributed, LSTM, Dense, Flatten, Dropout, Concatenate, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import cv2
import os
from tqdm import tqdm

from utils.logistic import run_logistic
from utils.visualizations import visualize

base_path = 'dataset/pediatric_echo_avi/A4C'

metadata = pd.read_csv(f'{base_path}/FileList.csv')
volume_tracings = pd.read_csv(f'{base_path}/VolumeTracings.csv')

print(metadata.describe())

# visualize(metadata, volume_tracings, base_path)
# run_logistic(metadata, volume_tracings)

frame_shape = (224, 224, 3)  # Frame dimensions after resizing
sequence_length = 30  # Number of frames per video

scaler = StandardScaler()
numerical_features = ['Age', 'Weight', 'Height']
metadata[numerical_features] = scaler.fit_transform(metadata[numerical_features])

volume_features = volume_tracings.groupby('FileName').agg({
    'X': ['mean', 'std', 'min', 'max'],
    'Y': ['mean', 'std', 'min', 'max']
}).reset_index()

volume_features.columns = ['_'.join(col).strip() for col in volume_features.columns.values]

data = pd.merge(metadata, volume_features, left_on='FileName', right_on='FileName_', how='inner')

data = data.sample(frac=0.001, random_state=42).reset_index(drop=True)

print(data.describe())
print(len(data))

base_model = VGG16(weights='imagenet', include_top=False, input_shape=frame_shape)
feature_extractor = Model(inputs=base_model.input, outputs=base_model.output)


def extract_video_features(video_path, interval=1):
    cap = cv2.VideoCapture(video_path)
    frame_features = []
    count = 0
    while cap.isOpened() and len(frame_features) < sequence_length:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame = cv2.resize(frame, (224, 224))
            frame = np.expand_dims(frame, axis=0)
            frame = tf.keras.applications.vgg16.preprocess_input(frame)
            features = feature_extractor.predict(frame)
            frame_features.append(features[0])
        count += 1
    cap.release()
    while len(frame_features) < sequence_length:
        frame_features.append(np.zeros_like(frame_features[0]))
    return np.array(frame_features)


video_dir = f'{base_path}/Videos'
video_features = []
ef_values = []

for index, row in tqdm(data.iterrows(), total=data.shape[0]):
    filename = row['FileName']
    video_path = os.path.join(video_dir, filename)

    if not os.path.exists(video_path):
        continue

    video_feat = extract_video_features(video_path)
    print(video_feat)
    video_features.append(video_feat)
    ef_values.append(row['EF'])

video_features = np.array(video_features)
ef_values = np.array(ef_values)

demographic_features = data[numerical_features + list(volume_features.columns[1:])].values
combined_features = [video_features, demographic_features]

print(len(video_features), len(demographic_features))

X_video_train, X_video_val, X_demo_train, X_demo_val, y_train, y_val = train_test_split(
    video_features, demographic_features, ef_values, test_size=0.2, random_state=42
)

# After splitting, combined_features for training and validation
X_train = [X_video_train, X_demo_train]
X_val = [X_video_val, X_demo_val]

video_input = Input(shape=(sequence_length, 7, 7, 512))
flattened_features = TimeDistributed(Flatten())(video_input)
video_lstm = LSTM(256, return_sequences=True)(flattened_features)
video_lstm = Dropout(0.5)(video_lstm)
video_lstm = LSTM(128)(video_lstm)
video_lstm = Dropout(0.5)(video_lstm)

demographic_input = Input(shape=(demographic_features.shape[1],))
demographic_dense = Dense(64, activation='relu')(demographic_input)

combined = Concatenate()([video_lstm, demographic_dense])
combined = Dense(64, activation='relu')(combined)
output = Dense(1, activation='linear')(combined)

model = Model(inputs=[video_input, demographic_input], outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=8)

loss, mae = model.evaluate([X_val[0], X_val[1]], y_val)
print(f'Validation Mean Absolute Error: {mae:.2f}')
