import numpy as np
import pandas as pd

from scipy.stats import boxcox
from sklearn.decomposition import PCA


def scale(X_train, X_test, scale_type='minmax', axis=None, use_boxcox=False, boxcox_axis=None, use_pandas=False):
	"""
		Scale the data using either minmax of standard scaling procedure

		Parameters:
		-----------
		X_train, X_test : np.array (n, n_features)
		Arrays of input features where each row should be a single feature set
		
		scale_type: {'minmax', 'standard'}
		Scaler to use (default: minmax
		
		axis: {1, 0, None}
		Axis over which to scale. (default: None - get single value from array)
		
		use_boxcox: bool
		Where as to use the boxcox transformation before scaling
		
		boxcox_axis: {0, None}
		
		verbose: bool

		Returns
		-------
		X_train, X_test : np.array (n, n_in)
			Scaled and reduced features
	"""
	if isinstance(X_train, pd.DataFrame) and use_pandas:
		train_idx, test_idx = X_train.index, X_test.index
		X_train, X_test = X_train.values, X_test.values
	
	if use_boxcox:
		X_train, X_test = boxcox_transform(X_train, X_test, axis=boxcox_axis)
	
	if scale_type == 'minmax':
		min_ = np.min(X_train, axis=axis)
		max_ = np.max(X_train, axis=axis)
		diff = max_ - min_
		diff[diff == 0] = 1  # Avoid 0 for numerical stability
		X_train = (X_train - min_) / diff
		X_test = (X_test - min_) / diff
	elif scale == 'standard':
		mean = np.mean(X_train, axis=axis)
		std = np.std(X_train, axis=axis)
		X_train = (X_train - mean) / std
		X_test = (X_test - mean) / std
		
	if use_pandas:
		X_train, X_test = pd.DataFrame(X_train, index=train_idx), \
		                  pd.DataFrame(X_test, index=test_idx)
		
	return X_train, X_test


def boxcox_transform(X_train, X_test, axis=None):
	# Used to add before boxcox transformation to ensure all values are positive
	sv = 1
	if axis:
		for i in range(X_train.shape[1]):
			X_train[:, i], maxlog = boxcox(X_train[:, i] + sv)
			X_test = boxcox(X_test[:, i] + sv, maxlog)
	else:
		train_shape = X_train.shape
		test_shape = X_test.shape
		X_train, maxlog = boxcox(X_train.flatten() + sv)
		X_train = X_train.reshape(train_shape)
		X_test = boxcox(X_test.flatten() + sv, maxlog).reshape(test_shape)
		
	return X_train, X_test


def pca_transform(X_train, X_test, pca_components, use_pandas=False):
	
	if isinstance(X_train, pd.DataFrame) and use_pandas:
		train_idx, test_idx = X_train.index, X_test.index
		X_train, X_test = X_train.values, X_test.values
		
	if pca_components < 1:
		pca = PCA().fit(X_train)
		pca_components = np.where(np.cumsum(pca.explained_variance_ratio_) > pca_components)[0][0]
	pca = PCA(n_components=pca_components).fit(X_train)
	X_train, X_test = pca.transform(X_train), pca.transform(X_test)
	
	if use_pandas:
		X_train, X_test = pd.DataFrame(X_train, index=train_idx), \
		                  pd.DataFrame(X_test, index=test_idx)
		
	return X_train, X_test, pca_components