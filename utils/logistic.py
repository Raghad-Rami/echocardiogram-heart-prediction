from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


def run_logistic(metadata, volume_tracings):
    metadata = pd.get_dummies(metadata, columns=['Sex'], drop_first=True)

    metadata['EF_Binary'] = (metadata['EF'] < 50).astype(int)

    features = ['Age', 'Weight', 'Height']
    scaler = StandardScaler()
    metadata[features] = scaler.fit_transform(metadata[features])

    volume_features = volume_tracings.groupby('FileName').agg({
        'X': ['mean', 'std', 'min', 'max'],
        'Y': ['mean', 'std', 'min', 'max']
    }).reset_index()

    volume_features.columns = ['_'.join(col).strip() for col in volume_features.columns.values]

    data = pd.merge(metadata, volume_features, left_on='FileName', right_on='FileName_', how='inner')

    data = data.dropna()

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
