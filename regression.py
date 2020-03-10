import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from video_feature import get_video_features

features = get_video_features()
features['id'] = [int(i[:3]) for i in features.index]
features['n'] = [int(i[4:5]) for i in features.index]
features.set_index(['id', 'n'], inplace=True)
features.sort_index(inplace=True)
features = features.groupby('id').mean()

labels = pd.read_csv('data/dev_split_Depression_AVEC2017.csv')
labels.set_index('Participant_ID', inplace=True)

scaled = StandardScaler().fit_transform(features)
pca = PCA(n_components=15).fit_transform(scaled)
pca_scaled = MinMaxScaler().fit_transform(pca)


