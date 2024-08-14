import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 1. Data Loading and Preprocessing

def load_csv_data(folder):
    volume_tracings = pd.read_csv(os.path.join(folder, 'VolumeTracings.csv'))
    file_list = pd.read_csv(os.path.join(folder, 'FileList.csv'))
    return volume_tracings, file_list

def load_video_data(folder, filename, max_frames=100):
    video_path = os.path.join(folder, 'videos', filename)
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64))
        frames.append(frame)
    cap.release()
    
    # Pad or truncate to max_frames
    if len(frames) < max_frames:
        frames += [np.zeros((64, 64, 3))] * (max_frames - len(frames))
    elif len(frames) > max_frames:
        frames = frames[:max_frames]
    
    return np.array(frames)

# Load data from both folders
a4c_volume, a4c_file_list = load_csv_data('A4C')
psax_volume, psax_file_list = load_csv_data('PSAX')

# Combine data
file_list = pd.concat([a4c_file_list, psax_file_list]).reset_index(drop=True)
volume_tracings = pd.concat([a4c_volume, psax_volume]).reset_index(drop=True)

# 2. Feature Engineering
'''
def extract_features(volume_tracings, file_list):
    features = []
    for filename in file_list['FileName']:
        video_frames = volume_tracings[volume_tracings['FileName'] == filename]
        features.append([
            video_frames['X'].mean(),
            video_frames['X'].std(),
            video_frames['Y'].mean(),
            video_frames['Y'].std(),
            video_frames['Frame'].max()
        ])
    return np.array(features)

csv_features = extract_features(volume_tracings, file_list)
csv_features = np.hstack((csv_features, file_list[['sex', 'age', 'weight', 'height']].values))

# Encode sex
csv_features[:, 5] = (csv_features[:, 5] == 'M').astype(int)

# Normalize features
scaler = StandardScaler()
csv_features = scaler.fit_transform(csv_features)

# 3. PyTorch Dataset and Model

class EchoDataset(Dataset):
    def __init__(self, csv_features, video_files, labels, folder):
        self.csv_features = csv_features
        self.video_files = video_files
        self.labels = labels
        self.folder = folder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        csv_feature = torch.FloatTensor(self.csv_features[idx])
        video = load_video_data(self.folder, self.video_files[idx])
        video = torch.FloatTensor(video).permute(3, 0, 1, 2)  # Change to channel-first format
        label = torch.FloatTensor([self.labels[idx]])
        return csv_feature, video, label

class CombinedModel(nn.Module):
    def __init__(self, csv_shape, video_shape):
        super(CombinedModel, self).__init__()
        self.csv_net = nn.Sequential(
            nn.Linear(csv_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        self.video_net = nn.Sequential(
            nn.Conv3d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * (video_shape[1]//4) * (video_shape[2]//4) * (video_shape[3]//4), 64),
            nn.ReLU()
        )
        
        self.combined_net = nn.Sequential(
            nn.Linear(32 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, csv_input, video_input):
        csv_features = self.csv_net(csv_input)
        video_features = self.video_net(video_input)
        combined = torch.cat((csv_features, video_features), dim=1)
        return self.combined_net(combined)

# 4. Prepare Data and Train Model

# Split data
X_csv_train, X_csv_test, y_train, y_test, train_indices, test_indices = train_test_split(
    csv_features, file_list['EF'], range(len(file_list)), test_size=0.2, random_state=42)

# Prepare datasets
train_dataset = EchoDataset(X_csv_train, file_list['FileName'].iloc[train_indices], y_train, 'A4C')
test_dataset = EchoDataset(X_csv_test, file_list['FileName'].iloc[test_indices], y_test, 'A4C')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CombinedModel(X_csv_train.shape[1], (3, 100, 64, 64)).to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for csv_features, videos, labels in train_loader:
        csv_features, videos, labels = csv_features.to(device), videos.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(csv_features, videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. Evaluation
model.eval()
predictions = []
true_labels = []

with torch.no_grad():
    for csv_features, videos, labels in test_loader:
        csv_features, videos = csv_features.to(device), videos.to(device)
        outputs = model(csv_features, videos)
        predictions.extend(outputs.cpu().numpy())
        true_labels.extend(labels.numpy())

predictions = np.array(predictions).flatten()
true_labels = np.array(true_labels).flatten()

mse = mean_squared_error(true_labels, predictions)
r2 = r2_score(true_labels, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")
'''
