from __future__ import print_function
from conv_qsar.utils.parsing import input_to_bool
from conv_qsar.utils.parse_cfg import read_config
import conv_qsar.utils.reset_layers as reset_layers
import rdkit.Chem as Chem
import numpy as np
import datetime
import json
import sys
import os
import time
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from conv_qsar.main.core import build_model, train_model, save_model
from conv_qsar.main.test import test_model, test_embeddings_demo
from conv_qsar.main.data import get_data_full

if __name__ == '__main__':
	if len(sys.argv) < 2:
		print('Usage: {} "settings.cfg"'.format(sys.argv[0]))
		quit(1)

	# Load settings
	try:
		config = read_config(sys.argv[1])
	except:
		print('Could not read config file {}'.format(sys.argv[1]))
		quit(1)

	# Get model label
	try:
		fpath = config['IO']['model_fpath']
	except KeyError:
		print('Must specify model_fpath in IO in config')
		quit(1)

	###################################################################################
	### BUILD MODEL
	###################################################################################

	print('...building model')
	try:
		kwargs = config['ARCHITECTURE']
		del kwargs['__name__'] #  from configparser
		if 'batch_size' in config['TRAINING']:
			kwargs['padding'] = int(config['TRAINING']['batch_size']) > 1
		if 'embedding_size' in kwargs: 
			kwargs['embedding_size'] = int(kwargs['embedding_size'])
		if 'hidden' in kwargs: 
			kwargs['hidden'] = int(kwargs['hidden'])
		if 'hidden2' in kwargs:
			kwargs['hidden2'] = int(kwargs['hidden2'])
		if 'depth' in kwargs: 
			kwargs['depth'] = int(kwargs['depth'])
		if 'scale_output' in kwargs: 
			kwargs['scale_output'] = float(kwargs['scale_output'])
		if 'dr1' in kwargs:
			kwargs['dr1'] = float(kwargs['dr1'])
		if 'dr2' in kwargs:
			kwargs['dr2'] = float(kwargs['dr2'])
		if 'output_size' in kwargs:
			kwargs['output_size'] = int(kwargs['output_size'])
		if 'sum_after' in kwargs:
			kwargs['sum_after'] = input_to_bool(kwargs['sum_after'])
		 
		if 'molecular_attributes' in config['DATA']:
			kwargs['molecular_attributes'] = config['DATA']['molecular_attributes']

		model = build_model(**kwargs)
		print('...built untrained model')
	except KeyboardInterrupt:
		print('User cancelled model building')
		quit(1)

	###################################################################################
	### DEFINE DATA 
	###################################################################################

	data_kwargs = config['DATA']
	if '__name__' in data_kwargs:
		del data_kwargs['__name__'] #  from configparser
	if 'batch_size' in config['TRAINING']:
		data_kwargs['batch_size'] = int(config['TRAINING']['batch_size'])
	if 'use_fp' in config['ARCHITECTURE']:
		data_kwargs['use_fp'] = config['ARCHITECTURE']['use_fp']
	if 'shuffle_seed' in data_kwargs:
		data_kwargs['shuffle_seed'] = int(data_kwargs['shuffle_seed'])
	else:
		data_kwargs['shuffle_seed'] = int(time.time())
	if 'truncate_to' in data_kwargs:
		data_kwargs['truncate_to'] = int(data_kwargs['truncate_to'])
	if 'training_ratio' in data_kwargs:
		data_kwargs['training_ratio'] = float(data_kwargs['training_ratio'])
	if 'molecular_attributes' in data_kwargs: 
		data_kwargs['molecular_attributes'] = input_to_bool(data_kwargs['molecular_attributes'])
	
	# Force all data to be "training" data
	data_kwargs['data_split'] = 'ratio'
	data_kwargs['training_ratio'] = 1.0
	data_kwargs['data_label'] += '-test'
	data = get_data_full(**data_kwargs)

	# Unpack
	(train, val, test) = data
	# Unpack
	mols_train = train['mols']; y_train = train['y']; smiles_train = train['smiles']
	# mols_val   = val['mols'];   y_val   = val['y'];   smiles_val   = val['smiles']
	# mols_test  = test['mols'];  y_test  = test['y'];  smiles_test  = test['smiles']
	y_label = train['y_label']

	if type(y_train[0]) != type(0.0):
		num_targets = y_train[0].shape[-1]
	else:
		num_targets = 1

	y_train_pred = np.array([np.array([0.0 for t in range(num_targets)]) for z in mols_train])
	print(y_train_pred.shape)
	# y_val_pred = np.array([0 for z in mols_val])
	# y_test_pred = np.array([0 for z in mols_test])


	ref_fpath = fpath
	cv_folds = range(1, 6)
	for cv_fold in cv_folds:
		print('Using weights from CV fold {}'.format(cv_fold))
		fpath = ref_fpath.replace('<this_fold>', str(cv_fold))

		# Load weights
		weights_fpath = fpath + '.h5'
		model.load_weights(weights_fpath)

		for j in tqdm(range(len(mols_train))):
			single_mol_as_array = np.array(mols_train[j:j+1])
			single_y_as_array = np.array(y_train[j:j+1])
			spred = model.predict_on_batch(single_mol_as_array)
			if num_targets == 1:
				y_train_pred[j] += spred
			else:
				y_train_pred[j,:] += spred.flatten()

	# Now divide by the number of folds to average predictions
	y_train_pred = y_train_pred / float(len(cv_folds))

	def round3(x):
		return int(x * 1000) / 1000.0

	test_fpath = os.path.dirname(ref_fpath)
	def parity_plot(true, pred, set_label):
		if len(true) == 0:
			print('skipping parity plot for empty dataset')
			return

		try:
			# Trim it to recorded values (not NaN)

			true = np.array(true).flatten()
			pred = np.array(pred).flatten()

			pred = pred[~np.isnan(true)]
			true = true[~np.isnan(true)]

			true = np.array(true).flatten()
			pred = np.array(pred).flatten()

			# For TOX21
			from sklearn.metrics import roc_auc_score, roc_curve, auc
			roc_x, roc_y, _ = roc_curve(true, pred)
			AUC = roc_auc_score(true, pred)
			plt.figure()
			lw = 2
			plt.plot(roc_x, roc_y, color='darkorange',
				lw = lw, label = 'ROC curve (area = %0.3f)' % AUC)
			plt.plot([0, 1], [0, 1], color='navy', lw = lw, linestyle = '--')
			plt.xlim([0.0, 1.0])
			plt.ylim([0.0, 1.05])
			plt.xlabel('False Positive Rate')
			plt.ylabel('True Positive Rate')
			plt.title('ROC for {}'.format(set_label))
			plt.legend(loc = "lower right")
			plt.savefig(os.path.join(test_fpath, ' {} ROC.png'.format(set_label)), bbox_inches = 'tight')
			plt.clf()

		except Exception as e:
			print(e)

	# Create plots for datasets
	if num_targets != 1:
		for i in range(num_targets):
				parity_plot([x[i] for x in y_train], [x[i] for x in y_train_pred], 'leaderboard set (consensus) - ' + y_label[i])
	else:
		parity_plot(y_train, y_train_pred, 'leaderboard set (consensus)')
