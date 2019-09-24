# -*- coding: utf-8 -*-
# Project: ForecastingStockPrice_thesis
# Created at: 21:50
import asyncio
import itertools
import time as t

import numpy as np
import pandas as pd
from pymongo import MongoClient
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import pacf, acf
from tqdm import tqdm_notebook as tqdm

from Models.Arima_Ann import HybridModel, AnnModel, ArimaModel
from Models.Lstm_geo_hybrid import LSTM_GBM
from src.config_tickets import ticket_lst
from src.scraping import WebScraping

__author__ = 'phuongnt18'
__email__ = 'phuongnt18@vng.com.vn'


def calculate_acf(time_series, lag=20, alpha=0.05):
	x = time_series.values
	acf_value, confint = acf(x, nlags=lag, alpha=alpha)
	confint_lower = confint[:, 0] - acf_value
	confint_upper = confint[:, 1] - acf_value
	return acf_value, confint_upper, confint_lower


def calculate_pacf(time_series, lag=20, alpha=0.05):
	x = time_series.values
	pacf_value, confint = pacf(x, nlags=lag, alpha=alpha)
	confint_lower = confint[:, 0] - pacf_value
	confint_upper = confint[:, 1] - pacf_value
	return pacf_value, confint_upper, confint_lower


def mean_absolute_percentage_error(y_true, y_pred):
    mean = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    # std = np.std(np.abs((y_true - y_pred) / y_true) * 100)
    return mean


def root_mean_squared_error(y_true, y_pred):
    mean = np.sqrt(np.mean((y_true - y_pred) ** 2))
    std = np.sqrt(np.std((y_true - y_pred) ** 2))
    return mean, std


def evaluation(validation):
	mse = mean_squared_error(validation['y'], validation['yhat'])
	std_mse = np.std((validation['y'] - validation['yhat']) ** 2)
	rmse, std_rmse = root_mean_squared_error(validation['y'], validation['yhat'])
	mae = mean_absolute_error(validation['y'], validation['yhat'])
	std_mae = np.std(np.abs(validation['y'] - validation['yhat']))
	mape, std_mape = mean_absolute_percentage_error(validation['y'], validation['yhat'])
	result = {
		'mse': mse,
		'std_mse': std_mse,
		'rmse': rmse,
		'std_rmse': std_rmse,
		'mae': mae,
		'std_mae': std_mae,
		'mape': mape,
		'std_mape': std_mape,
	}
	return result


def absolute_percentage_error(y, yhat):
	return np.abs((y - yhat)/y)*100


def run_model_with_parameters(train: pd.Series, test: pd.Series,
                              model_selection='ARIMA', order=(2, 1, 2),
                              lag=1, hidden_layers=(4, 3),
                              window_size=10):
	# split data
	# time_series = time_series.sort_index()
	# time_series = time_series.drop_duplicates()
	# size = len(time_series)
	# train_start = int(start_train * size)
	# train_end = int(end_train * size)
	# train, test = time_series[train_start:train_end], time_series[train_end:]

	# Run model
	result = {}
	model = None
	insample_data = None
	try:
		if model_selection == 'ARIMA':
			model = ArimaModel(train, order=order)
			insample_data = train[1:]
			result['order'] = order
		elif model_selection == 'ANN':
			model = AnnModel(train, lag=lag, hidden_layers=hidden_layers)
			insample_data = train[lag + 1:]
			result['lag'] = lag
			result['hidden_layers'] = hidden_layers
		elif model_selection == 'Hybrid':
			model = HybridModel(train, order=order, lag=lag, hidden_layers=hidden_layers)
			insample_data = train[lag + 1:]
			result['order'] = order
			result['lag'] = lag
			result['hidden_layers'] = hidden_layers
		# elif model_selection == 'LSTM_GBM':
		# 	print('Run model LSTM')
		# 	model = LSTM_GBM(train, window_size=window_size, lags=lag, verbose=True)
		# 	insample_data = train[window_size + lag:]
		# 	result['window_size'] = window_size
		# 	result['lag'] = lag

		# Fit model
		start = t.time()
		model.fit()
		running_time = t.time() - start

		# Evaluation
		# train_validate = pd.DataFrame()
		# train_validate['y'] = insample_data
		# train_validate['yhat'] = model.get_insample_prediction()
		# train_result = evaluation(train_validate)
		# result['train_evaluation'] = train_result

		test_validate = pd.DataFrame()
		test_validate['y'] = test
		test_validate['yhat'] = model.validate(test)
		test_result = absolute_percentage_error(test_validate['y'], test_validate['yhat'])
		result['yhat'] = test_validate['yhat']
		result['test_evaluation'] = test_result
		result['mape'] = mean_absolute_percentage_error(test_validate['y'], test_validate['yhat'])
		result['status'] = True
		result['model_name'] = model_selection
		result['model'] = model
		# result['training_time'] = running_time
		# if model_selection == 'LSTM_GBM':
		# 	df_drift = pd.DataFrame()
		# 	df_drift['y'] = model.target_drift
		# 	df_drift['yhat'] = model.pred_drift
		# 	df_volatility = pd.DataFrame()
		# 	df_volatility['y'] = model.target_volatility
		# 	df_volatility['yhat'] = model.pred_volatility
		# 	result['drift_validation'] = df_drift
		# 	result['volatility_validation'] = df_volatility
	except Exception as e:
		print(e)
		result['status'] = False

	return result


def gen_order(p, d, q):
	return list(itertools.product(p, d, q))


def gen_ann(lags, hl):
	hl1 = hl
	hl2 = [int(i / 2) for i in hl]
	return list(itertools.product(lags, hl1, hl2))


def gen_lstm(window_size, lags):
	return list(itertools.product(window_size, lags))


def choose_model(lst_result):
	mape = lst_result[0]['mape']
	result_selection = lst_result[0]
	for result in lst_result[1:]:
		if mape > result['mape']:
			mape = result['mape']
			result_selection = result
	return result_selection


def run_model_without_parameters(train: pd.Series, test: pd.Series, model_selection='ARIMA',
                                 p=range(0, 6), d=range(0, 2), q=range(0, 6),
                                 lags=range(1, 4), hl=range(3, 8), window_size=range(5, 11)):
	# Generate parameters
	lst_order = gen_order(p, d, q)
	lst_ann_param = gen_ann(lags, hl)
	lst_lstm = gen_lstm(window_size, lags)

	# Run model
	lst_result = list()
	if model_selection == 'ARIMA':
		for order in tqdm(lst_order, desc='ARIMA'):
			result = run_model_with_parameters(train, test, model_selection=model_selection,
			                                   order=order)
			if result['status']:
				result['params'] = order
				lst_result.append(result)
	elif model_selection == 'ANN':
		for ann_param in tqdm(lst_ann_param, desc='ANN'):
			result = run_model_with_parameters(train, test, model_selection=model_selection,
			                                   lag=ann_param[0], hidden_layers=ann_param[1:])
			if result['status']:
				result['params'] = ann_param
				lst_result.append(result)
	elif model_selection == 'Hybrid':
		# Choose the best ARIMA model
		lst_arima_result = list()
		for order in tqdm(lst_order, desc='Hybrid ARIMA'):
			result = run_model_with_parameters(train, test, model_selection='ARIMA', order=order)
			if result['status']:
				lst_arima_result.append(result)
		_result = choose_model(lst_arima_result)
		chosen_order = _result['order']

		# Choose the best ANN model
		for ann_param in tqdm(lst_ann_param, desc='Hybrid ANN'):
			result = run_model_with_parameters(train, test, model_selection=model_selection,
			                                   lag=ann_param[0], hidden_layers=ann_param[1:],
			                                   order=chosen_order)
			if result['status']:
				result['params'] = (chosen_order, ann_param)
				lst_result.append(result)
	# elif model_selection == 'LSTM+GBM':
	# 	for lstm_param in lst_lstm:
	# 		result = run_model_with_parameters(train, test, model_selection='LSTM+GBM',
	# 		                                   window_size=lstm_param[0], lag=lstm_param[1])
	# 		if result['status']:
	# 			lst_result.append(result)

	return lst_result


def request_2_website():
	options = Options()
	options.add_argument("--headless")
	prefs = {"profile.managed_default_content_settings.images": 2}
	options.add_experimental_option("prefs", prefs)
	drivers = list()
	print('Requesting...')
	for ticket in ticket_lst:
		driver = webdriver.Chrome(options=options)
		driver.get(ticket['url'])
		drivers.append(driver)
	print('Requesting completed.')
	return drivers


def update_database(client: MongoClient, years=5):
	driver_lst = request_2_website()
	scraper = WebScraping(driver_lst=driver_lst, dbClient=client, verbose=True)
	scraper.scrape_historical_data(years=years)


def get_real_time_data(client: MongoClient, on=True):
	print('blah blah')
	driver_lst = request_2_website()
	scraper = WebScraping(driver_lst=driver_lst, dbClient=client, verbose=True)

	loop = asyncio.get_event_loop()
	asyncio.ensure_future(scraper.start_scraping())
	if on:
		loop.run_forever()
	else:
		loop.stop()
