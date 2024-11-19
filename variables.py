import pandas as pd

data = pd.read_csv('preprocessed_data.csv')

labels = data.Depression
data = data.drop(columns=['Depression'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, labels, train_size=0.70, random_state=5)

__all__ = ['data', 'labels', 'X_train', 'X_test', 'y_train', 'y_test']