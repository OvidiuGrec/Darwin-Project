from sklearn.metrics import mean_squared_error, mean_absolute_error


def avec_metrics(y_true, y_pred):
	mae = mean_absolute_error(y_true, y_pred)
	rmse = mean_squared_error(y_true, y_pred, squared=False)
	return mae, rmse