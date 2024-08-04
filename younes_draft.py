import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('dataset/pediatric_echo_avi/A4C/FileList.csv')

print(df.describe())

sns.histplot(df['Age'], bins=20, kde=True)
plt.title('Age Distribution')
plt.show()
