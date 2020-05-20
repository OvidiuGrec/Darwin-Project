import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools

from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix

from helper import to_classes


def run_validation(y_true, y_pred, binary=False):
	
	y_pred = y_pred.groupby(level=0).mean()
	y_true = y_true.groupby(level=0).mean()
	per_patient_plot(y_true.copy(), y_pred.copy())
	y_true = y_true.values.flatten()
	y_pred = y_pred.values.flatten()
	error_plot(y_true, y_pred)
	plot_confusion_matrix(y_true.copy(), y_pred.copy(), binary=binary)
	

def avec_metrics(y_true, y_pred):
	y_pred = y_pred.groupby(level=0).mean()
	y_true = y_true.groupby(level=0).mean()
	mae = mean_absolute_error(y_true, y_pred)
	rmse = mean_squared_error(y_true, y_pred, squared=False)
	return mae, rmse


def per_patient_plot(y_true, y_pred):
	y_true.sort_values(by=0, inplace=True)
	y_pred = y_pred.reindex(y_true.index)
	x = np.arange(y_true.shape[0])
	fig, ax = plt.subplots(figsize=(10, 6))
	ax.set(title="Predicted vs True BDI-II score in Dev set",
		   xlabel="Patient Number in Dev Set",
		   ylabel="BDI-II")
	ax.stem(x, y_true, label='true', linefmt='r--', markerfmt='ro', basefmt=' ', use_line_collection=True)
	ax.stem(x, y_pred, label='pred', linefmt='b-', markerfmt='bx', basefmt=' ', use_line_collection=True)
	plt.xticks(x, y_true.index, rotation=70)
	ax.legend()
	plt.show()
	
	
def error_plot(y_true, y_pred):
	
	fig, ax = plt.subplots(2, figsize=(10, 12))
	ax[0].set_title('Distribution and Spread of Error')
	ax[0].hist(absolute_error(y_true, y_pred), bins=15)
	ax[1].scatter(y_true, y_pred)
	ax[1].plot(y_true, y_true)
	plt.show()
	
	
def plot_confusion_matrix(y_true, y_pred, binary=False):
	
	cm = confusion_matrix(to_classes(y_true, binary=binary), to_classes(y_pred, binary=binary))
	cm = np.flipud(cm)
	accuracy = np.trace(cm) / np.sum(cm).astype('float')
	misclass = 1 - accuracy
	
	if binary:
		labels = ['no dep', 'dep']
	else:
		labels = ['normal', 'mild', 'border', 'moderate', 'severe', 'extreme']

	cmap = plt.get_cmap('Blues')

	plt.figure(figsize=(8, 6))
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title('Confusion matrix')
	plt.colorbar()

	if labels is not None:
		tick_marks = np.arange(len(labels))
		plt.xticks(tick_marks, labels, rotation=45)
		plt.yticks(tick_marks, labels[::-1])

	thresh = cm.max() / 2
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, "{:,}".format(cm[i, j]),
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel(f'Predicted label\n accuracy={accuracy:.2f}; miss_class={misclass:.2f}')
	plt.show()
	

def absolute_error(y_true, y_pred):
	return np.abs(y_true - y_pred)