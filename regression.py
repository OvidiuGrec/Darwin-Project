import pandas as pd
import os

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from video_feature import get_video_features

features = get_video_features()
features['id'] = [int(i[:3]) for i in features.index]
features['n'] = [int(i[4:5]) for i in features.index]
features.set_index(['id', 'n'], inplace=True)
features.sort_index(inplace=True)
features = features.groupby('id').mean()

labels = pd.DataFrame(index=features.index, columns=['score'])
for file in os.listdir('data/labels/dev'):
    label = pd.read_csv(f'data/labels/dev/{file}')
    labels.loc[int(file[:3]), 'score'] = label.columns.values[0]

scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=15).fit_transform(scaled)
pca_scaled = MinMaxScaler().fit_transform(pca)

X_train, X_test, y_train, y_test = train_test_split(pca_scaled, labels, test_size=0.2)

model = LinearRegression().fit(X_train, y_train)
pred = model.predict(X_test)

mean_absolute_error(y_test, pred)


