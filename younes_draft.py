import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

base_path = 'dataset/pediatric_echo_avi/A4C'

df = pd.read_csv(f'{base_path}/FileList.csv')

print(df.describe())

# sns.histplot(df['Age'], bins=20, kde=True)
# plt.title('Age Distribution')
# plt.show()


tracings = pd.read_csv(f'{base_path}/VolumeTracings.csv')

filename = 'CR32a7555-CR32a7582-000039.avi'
frame_number = 39
tracings_for_frame = tracings[(tracings['FileName'] == filename) & (tracings['Frame'] == str(frame_number))]

cap = cv2.VideoCapture(f'{base_path}/Videos/{filename}')
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
ret, frame = cap.read()

plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
plt.scatter(tracings_for_frame['X'], tracings_for_frame['Y'], color='red', s=5)
plt.title(f'Frame {frame_number} Tracings')
plt.show()

cap.release()
