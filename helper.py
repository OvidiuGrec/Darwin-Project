import numpy as np
import pandas as pd
import pickle
import os

from pathlib import Path


def to_classes(y, binary=False):
	for i in range(len(y)):
		bdi = y[i]
		
		if binary:
			if bdi < 20:
				y[i] = 0
			else:
				y[i] = 1
		else:
			if bdi <= 10:
				y[i] = 0  # normal
			elif 10 < bdi <= 16:
				y[i] = 1  # mild mood
			elif 16 < bdi <= 20:
				y[i] = 2  # (borderline) clinical depression
			elif 20 < bdi <= 30:
				y[i] = 3  # moderate depression
			elif 30 < bdi <= 40:
				y[i] = 4  # sever depression
			elif bdi > 40:
				y[i] = 5  # extreme
	return y


def save_to_file(folder, file_name, data):
	if not os.path.exists(folder):
		os.makedirs(folder)

	if isinstance(data, pd.DataFrame):
		data.to_pickle(Path(folder, file_name))
	else:
		with open(Path(folder, file_name), 'wb') as f:
			pickle.dump(data, f)


def load_from_file(path):
	if os.path.isfile(path):
		try:
			return pd.read_pickle(path)
		except:
			with open(path, 'rb') as f:
				return pickle.load(f)
	else:
		raise FileNotFoundError(f'{path} does not exist')


def shift_n_days(arr, start, steps, reverse=False):
	initial_arr = arr.copy()
	for i in range(start, steps):
		if reverse:
			i = -i
		arr = np.hstack([arr, shift_day(initial_arr, i)])

	return arr


def shift_day(arr, num, fill_value=np.nan):
	result = np.empty_like(arr)
	if num > 0:
		result[:num] = fill_value
		result[num:] = arr[:-num]
	elif num < 0:
		result[num:] = fill_value
		result[:num] = arr[-num:]
	else:
		result[:] = arr
	return result