from __future__ import print_function
from conv_qsar_fast.utils.parsing import input_to_bool
from conv_qsar_fast.utils.parse_cfg import read_config
import conv_qsar_fast.utils.reset_layers as reset_layers
import rdkit.Chem as Chem
import cPickle as pickle
import numpy as np
import datetime
import json
import sys
import os
import sys
import time

from conv_qsar_fast.main.core import build_model, train_model, save_model
from conv_qsar_fast.main.test import test_model, test_embeddings_demo
from conv_qsar_fast.main.data import get_data_full

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

	if 'cv_folds' in data_kwargs:
		try:
			os.makedirs(os.path.dirname(fpath))
		except: # folder exists
			pass

		# Look for seed from file
		seed_file = os.path.join(os.path.dirname(fpath), 'data_seed.txt')
		if os.path.isfile(seed_file):
			with open(seed_file, 'r') as fid: 
				seed = int(fid.read().strip())
				print('reloaded seed: {}'.format(seed))
				data_kwargs['shuffle_seed'] = seed 
		else:
			with open(seed_file, 'w') as fid: 
				fid.write('{}'.format(data_kwargs['shuffle_seed']))

		# Define CV folds
		if '<this_fold>' in data_kwargs['cv_folds']:
			cv_folds = data_kwargs['cv_folds']
			total_folds = int(cv_folds.split('/')[1])
			all_cv_folds = ['{}/{}'.format(i + 1, total_folds) for i in range(total_folds)]
		else:
			all_cv_folds = [data_kwargs['cv_folds']]
	else:
		all_cv_folds = ['1/1']

	
	##################################################################################
	### BUILD MODEL PARAMS
	###################################################################################

	print('...building model')
	build_kwargs = config['ARCHITECTURE']
	if '__name__' in build_kwargs: del build_kwargs['__name__'] #  from configparser
	if 'batch_size' in config['TRAINING']:
		build_kwargs['padding'] = int(config['TRAINING']['batch_size']) > 1
	if 'embedding_size' in build_kwargs: 
		build_kwargs['embedding_size'] = int(build_kwargs['embedding_size'])
	if 'hidden' in build_kwargs: 
		build_kwargs['hidden'] = int(build_kwargs['hidden'])
	if 'hidden2' in build_kwargs:
		build_kwargs['hidden2'] = int(build_kwargs['hidden2'])
	if 'depth' in build_kwargs: 
		build_kwargs['depth'] = int(build_kwargs['depth'])
	if 'scale_output' in build_kwargs: 
		build_kwargs['scale_output'] = float(build_kwargs['scale_output'])
	if 'dr1' in build_kwargs:
		build_kwargs['dr1'] = float(build_kwargs['dr1'])
	if 'dr2' in build_kwargs:
		build_kwargs['dr2'] = float(build_kwargs['dr2'])
	if 'output_size' in build_kwargs:
		build_kwargs['output_size'] = int(build_kwargs['output_size'])
	if 'sum_after' in build_kwargs:
		build_kwargs['sum_after'] = input_to_bool(build_kwargs['sum_after'])
	if 'molecular_attributes' in config['DATA']:
		build_kwargs['molecular_attributes'] = config['DATA']['molecular_attributes']


	# Iterate through all folds
	ref_fpath = fpath

	# Get ready for multiprocessing 
	from multiprocessing import Pool 

	def do_a_single_fold(cv_fold):

		#sys.stdout = open(str(os.getpid()) + ".out", "w", buffering=0)
		#sys.stderr = open(str(os.getpid()) + "err.out", "w", buffering=0)
		this_data_kwargs = data_kwargs.copy()
		print('Using CV fold {}'.format(cv_fold))
		this_data_kwargs['cv_folds'] = cv_fold

		this_data_kwargs['training_ratio'] = 1.0 # Keep all data in training / testing, NO validation defined here
		data = get_data_full(**this_data_kwargs)

		# Special case for tox21 - need to combine two datasets for training/testing
		if '-traintest' in data_kwargs['data_label']:
			(train, val, test) = data
			# Get eval data if training on traintest
			this_data_kwargs['data_label'] = data_kwargs['data_label'].replace('-traintest', '-eval')
			testonly_data = get_data_full(**this_data_kwargs)
			(testonly_train, testonly_val, testonly_test) = testonly_data
			# Repackage, replacing the "val" data with the test data
			data = (train, testonly_train, testonly_train)


		from collections import defaultdict
		all_conditions_valMSE = defaultdict(float)
		all_conditions_testMSE = defaultdict(float) # just for debugging code, correlating val and test performance
		all_conditions = dict()

		## Look for record of completed runs
		fpath = ref_fpath.replace('<this_fold>', cv_fold.split('/')[0])
		all_conditions_valMSE_fpath = fpath + '_valMSE.pickle'
		all_conditions_testMSE_fpath = fpath + '_testMSE.pickle'
		completed_runs_fpath = fpath + '_completed.pickle'
		if os.path.isfile(all_conditions_valMSE_fpath):
			with open(all_conditions_valMSE_fpath, 'rb') as fid:
				all_conditions_valMSE_i = pickle.load(fid)
				all_conditions_valMSE = defaultdict(float, all_conditions_valMSE_i)
		if os.path.isfile(all_conditions_testMSE_fpath):
			with open(all_conditions_testMSE_fpath, 'rb') as fid:
				all_conditions_testMSE_i = pickle.load(fid)
				all_conditions_testMSE = defaultdict(float, all_conditions_testMSE_i)
		if os.path.isfile(completed_runs_fpath):
			with open(completed_runs_fpath, 'rb') as fid: 
				completed_runs = pickle.load(fid)
				print('Looks like {} runs were already done!'.format(len(completed_runs)))
		else:
			completed_runs = []
		

		from copy import deepcopy
		NUM_REPLICATES_EACH_SETTING = 3
		for replicate_num in range(NUM_REPLICATES_EACH_SETTING):
			print('REPLICATE {}'.format(replicate_num))

			# Split off a random 20% of this data for evaluation for testing hyperparameters

			this_data = deepcopy(data)
			(train, val, test) = this_data
			mols_train = train['mols']; y_train = train['y']; smiles_train = train['smiles']

			indices = range(len(mols_train))
			np.random.seed(data_kwargs['shuffle_seed'] + 1 + replicate_num * 100)
			np.random.shuffle(indices)
			cutoff = int(0.8 * len(indices))

			val['mols']   = [train['mols'][i] for i in indices[cutoff:]]
			val['y']      = [train['y'][i] for i in indices[cutoff:]]
			val['smiles'] = [train['smiles'][i] for i in indices[cutoff:]]

			test['mols']   = val['mols']
			test['y']      = val['y']
			test['smiles'] = val['smiles']

			train['mols']   = [train['mols'][i] for i in indices[:cutoff]]
			train['y']      = [train['y'][i] for i in indices[:cutoff]]
			train['smiles'] = [train['smiles'][i] for i in indices[:cutoff]]

			this_data = (train, val, test)

			# Define possible hyperparameters
			from itertools import product
			hyperparam_combos = product(
				[2, 3, 4, 5], # depth
				[32, 64, 128], # inner rep size
				[0.003, 0.001, 0.0003, 0.0001, 0.00003], # learning rates
				[0.0], # dropout
			)
			# Get random training/validation split
			for (depth, inner_rep, lr, dr) in hyperparam_combos:
				specific_tag = 'depth{}_innersize{}_lr{}_dr{}_replicate{}'.format(depth, inner_rep, lr, dr, replicate_num)

				print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				print(specific_tag)
				print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
				print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

				fpath = ref_fpath.replace('<this_fold>', cv_fold.split('/')[0])

				conditionlabel = 'depth{}_innersize{}_lr{}_dr{}'.format(depth, inner_rep, lr, dr)
				condition = (depth, inner_rep, lr, dr)
				all_conditions[conditionlabel] = condition

				## Now skip if we have already done this
				if specific_tag in completed_runs:
					print('Already done! No need to run this setting again')
					continue

				fpath += specific_tag

				build_kwargs['depth'] = depth
				build_kwargs['dr1'] = dr 
				build_kwargs['dr2'] = dr 
				build_kwargs['inner_rep'] = inner_rep 
				build_kwargs['lr'] = lr

				model = build_model(**build_kwargs)
				print('...built untrained model')

				# Train model
				try:
					print('...training model')
					kwargs = config['TRAINING']
					if '__name__' in kwargs:
						del kwargs['__name__'] #  from configparser
					if 'nb_epoch' in kwargs:
						kwargs['nb_epoch'] = int(kwargs['nb_epoch'])
					if 'batch_size' in kwargs:
						kwargs['batch_size'] = int(kwargs['batch_size'])
					if 'patience' in kwargs:
						kwargs['patience'] = int(kwargs['patience'])
					if 'lr_func' in kwargs:
						del kwargs['lr_func']
					(model, loss, val_loss) = train_model(model, this_data, **kwargs)
					print('...trained model')
				except KeyboardInterrupt:
					pass

				###################################################################################
				### SAVE MODEL
				###################################################################################

				# Get the current time
				tstamp = datetime.datetime.utcnow().strftime('%m-%d-%Y_%H-%M')
				print('...saving model')
				save_model(model, 
					loss,
					val_loss,
					fpath = fpath,
					config = config, 
					tstamp = tstamp)
				print('...saved model')

				###################################################################################
				### TEST MODEL
				###################################################################################

				print('...testing model')
				test_valMSE = test_model(model, this_data, fpath, tstamp = tstamp + '_val',
					batch_size = int(config['TRAINING']['batch_size']),
					return_test_MSE = True)
				print('...tested model')

				all_conditions_valMSE[conditionlabel] += test_valMSE

				print('...testing model')
				test_MSE = test_model(model, data, fpath, tstamp = tstamp + '_test',
					batch_size = int(config['TRAINING']['batch_size']),
					return_test_MSE = True)
				print('...tested model')
				all_conditions_testMSE[conditionlabel] += test_MSE


				completed_runs.append(specific_tag)
				with open(all_conditions_valMSE_fpath, 'wb') as fid: 
					pickle.dump(all_conditions_valMSE, fid)
				with open(all_conditions_testMSE_fpath, 'wb') as fid: 
					pickle.dump(all_conditions_testMSE, fid)
				with open(completed_runs_fpath, 'wb') as fid: 
					pickle.dump(completed_runs, fid)
				print('Saved')

				print(all_conditions_valMSE)
				print(all_conditions_testMSE)


		
		##############################################
		##############################################
		# Get best hyperparams now
		best_MSE = min(dict(all_conditions_valMSE).values())
		for conditionlabel in all_conditions_valMSE.keys():
			if all_conditions_valMSE[conditionlabel] == best_MSE:
				(depth, inner_rep, lr, dr) = all_conditions[conditionlabel]
				break

		print('~~~~ BEST CONDITIONS: ')
		print(conditionlabel)

		fpath = ref_fpath.replace('<this_fold>', cv_fold.split('/')[0])
		fpath += 'BEST_depth{}_innersize{}_lr{}_dr{}'.format(depth, inner_rep, lr, dr)

		with open(fpath + '.txt', 'w') as fid:
			for conditionlabel in all_conditions_valMSE.keys():
				fid.write('{} \t {} \t {}\n'.format(
					conditionlabel, 
					all_conditions_valMSE[conditionlabel] / float(NUM_REPLICATES_EACH_SETTING),
					all_conditions_testMSE[conditionlabel] / float(NUM_REPLICATES_EACH_SETTING),

				))

		build_kwargs['depth'] = depth
		build_kwargs['dr1'] = dr 
		build_kwargs['dr2'] = dr 
		build_kwargs['inner_rep'] = inner_rep 
		build_kwargs['lr'] = lr

		model = build_model(**build_kwargs)
		print('...built untrained model')

		# Train model
		try:
			print('...training model')
			kwargs = config['TRAINING']
			if '__name__' in kwargs:
				del kwargs['__name__'] #  from configparser
			if 'nb_epoch' in kwargs:
				kwargs['nb_epoch'] = int(kwargs['nb_epoch'])
			if 'batch_size' in kwargs:
				kwargs['batch_size'] = int(kwargs['batch_size'])
			if 'patience' in kwargs:
				kwargs['patience'] = int(kwargs['patience'])
			(model, loss, val_loss) = train_model(model, data, **kwargs)
			print('...trained model')
		except KeyboardInterrupt:
			pass

		###################################################################################
		### SAVE MODEL
		###################################################################################

		# Get the current time
		tstamp = datetime.datetime.utcnow().strftime('%m-%d-%Y_%H-%M')
		print('...saving model')
		save_model(model, 
			loss,
			val_loss,
			fpath = fpath,
			config = config, 
			tstamp = tstamp)
		print('...saved model')

		###################################################################################
		### TEST MODEL
		###################################################################################

		print('...testing model')
		test_model(model, data, fpath, tstamp = tstamp,
			batch_size = int(config['TRAINING']['batch_size']))
		print('...tested model')


	p = Pool(5)
	print('Starting multiprocessing pool')
	p.map(do_a_single_fold, all_cv_folds) # only do first one for now
