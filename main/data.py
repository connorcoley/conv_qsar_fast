from conv_qsar_fast.utils.neural_fp import *
import rdkit.Chem as Chem
import numpy as np
import os
import csv

def get_data_full(data_label = '', **kwargs):
	'''Wrapper for get_data_full, which allows for multiple datasets to
	be concatenated as a multi-target problem'''

	all_data = ()
	for data_label in data_label.split(','):
		data = get_data_one(data_label = data_label, **kwargs)
		if all_data:
			all_data = merge_data(all_data, data)
		else:
			all_data = data

	print('AFTER MERGING DATASETS...')
	print('# training: {}'.format(len(all_data[0]['y'])))
	print('# validation: {}'.format(len(all_data[1]['y'])))
	print('# testing: {}'.format(len(all_data[2]['y'])))

	return all_data

def merge_data(data1, data2):
	'''Combines two sets of data for multi-target prediction'''

	# Iterate over train, validate, test
	for i in range(3):

		if 'y' not in data2[i]: continue
		if len(data2[i]['y']) == 0: continue

		# Just append/extend mols, smiles, label
		data1[i]['mols'].extend(data2[i]['mols'])
		data1[i]['smiles'].extend(data2[i]['smiles'])

		# Not the first merge
		if type(data1[i]['y_label']) == type([]):
			data1[i]['y_label'].append(data2[i]['y_label'])
		else: # First merge
			data1[i]['y_label'] = [data1[i]['y_label'], data2[i]['y_label']]
		
		# Now y values
		example_y = data1[i]['y'][0]
		data1[i]['y'] = [np.append(x, np.nan) for x in data1[i]['y']]
		data1[i]['y'].extend([np.append(example_y * np.nan, x) for x in data2[i]['y']])

	return data1

def get_data_one(data_label = '', shuffle_seed = None, batch_size = 1, 
	data_split = 'cv', cv_folds = '1/1',	truncate_to = None, training_ratio = 0.9,
	molecular_attributes = False, use_fp = None):
	'''This is a helper script to read the data file and return
	the training and test data sets separately. This is to allow for an
	already-trained model to be evaluated using the test data (i.e., which
	we know it hasn't seen before)'''

	# Roots
	data_label = data_label.lower()
	data_froot = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

	###################################################################################
	### WHICH DATASET ARE WE TRYING TO USE?
	###################################################################################

	# Delaney solubility
	if data_label in ['delaney', 'delaney sol']:
		delimeter = ','
		dset = 'delaney'
		data_fpath = os.path.join(data_froot, 'Delaney2004.txt')
		smiles_index = 3
		y_index = 1
		def y_func(x): return x
		y_label = 'log10(aq sol (M))'

	# Abraham octanol set
	elif data_label in ['abraham', 'abraham sol', 'abraham_oct']:
		delimeter = ','
		dset = 'abraham'
		data_fpath = os.path.join(data_froot, 'AbrahamAcree2014_Octsol_partialSmiles.csv')
		smiles_index = 1
		y_index = 5
		def y_func(x): return x
		y_label = 'log10(octanol sol (M))'

	elif data_label in ['bradley_good', 'bradley']:
		delimeter = ','
		dset = 'bradley_good'
		data_fpath = os.path.join(data_froot, 'BradleyDoublePlusGoodMeltingPointDataset.csv')
		smiles_index = 2
		y_index = 3
		def y_func(x): return x
		y_label = 'Tm (deg C)'

	elif 'nr' in data_label or 'sr' in data_label:
		print('Assuming TOX21 data {}'.format(data_label))
		delimeter = '\t'
		dset = data_label
		data_fpath = os.path.join(data_froot, '{}.smiles'.format(data_label))
		smiles_index = 0
		y_index = 2
		def y_func(x): return x
		y_label = 'Active'

	elif 'tox21' == data_label:
		print('Assuming ALL TOX21 data')
		delimeter = '\t'
		dset = 'tox21'
		data_fpath = os.path.join(data_froot, 'tox21.smiles')
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']
		smiles_index = 0
		y_index = 2
		def y_func(x): return x

	elif 'tox21-test' == data_label:
		print('Assuming ALL TOX21 data, leaderboard test set')
		delimeter = '\t'
		dset = 'tox21-test'
		data_fpath = os.path.join(data_froot, 'tox21-test.smiles')
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']
		smiles_index = 0
		y_index = 2
		def y_func(x): return x
	
	elif 'tox21-eval' == data_label:
		print('Assuming ALL TOX21 data, eval set')
		delimeter = '\t'
		dset = 'tox21-test'
		data_fpath = os.path.join(data_froot, 'tox21-eval.smiles')
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']
		smiles_index = 0
		y_index = 2
		def y_func(x): return x

	elif 'tox21-traintest' == data_label:
		print('Assuming traintest TOX21 data')
		delimeter = '\t'
		dset = 'tox21-traintest'
		data_fpath = os.path.join(data_froot, 'tox21-traintest.smiles')
		y_label = ['nr-ahr','nr-ar','nr-ar-lbd','nr-aromatase','nr-er','nr-er-lbd',
				'nr-ppar-gamma','sr-are','sr-atad5','sr-hse','sr-mmpp','sr-p53']
		smiles_index = 0
		y_index = 2
		def y_func(x): return x

	# Other?
	else:
		print('Unrecognized data_label {}'.format(data_label))
		quit(1)


	###################################################################################
	### READ AND TRUNCATE DATA
	###################################################################################

	print('reading data...')
	data = []
	with open(data_fpath, 'r') as data_fid:
		reader = csv.reader(data_fid, delimiter = delimeter, quotechar = '"')
		for row in reader:
			data.append(row)
	print('done')
		
	# Truncate if necessary
	if truncate_to is not None:
		data = data[:truncate_to]
		print('truncated data to first {} samples'.format(truncate_to))

	# Get new shuffle seed if possible
	if shuffle_seed is not None:
		np.random.seed(shuffle_seed)
	
	###################################################################################
	### ITERATE THROUGH DATASET AND CREATE NECESSARY DATA LISTS
	###################################################################################

	smiles = []
	mols = []
	y = []
	print('processing data...')
	# Randomize
	np.random.shuffle(data)
	for i, row in enumerate(data):
		try:
			# Molecule first (most likely to fail)
			mol = Chem.MolFromSmiles(row[smiles_index], sanitize = False)
			Chem.SanitizeMol(mol)
			
			(mat_features, mat_adjacency, mat_specialbondtypes) = molToGraph(mol, molecular_attributes = molecular_attributes).dump_as_matrices()
			
			# Are we trying to use Morgan FPs?
			if use_fp == 'Morgan':
				mat_features = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,3,nBits=512,useFeatures=True))
				#print(mol_tensor)
			elif use_fp == 'Morgan2':
				mat_features = np.array(AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=512,useFeatures=True))
			elif type(use_fp) != type(None):
				print('Unrecognised use_FP option {}'.format(use_fp))

			if 'tox21' not in dset:
				this_y = y_func(float(row[y_index]))
			else: # get full TOX21 data
				this_y = np.array([float(x) for x in row[y_index:]])

			mols.append((mat_features, mat_adjacency, mat_specialbondtypes))

			y.append(this_y) # Measured log(solubility M/L)
			smiles.append(Chem.MolToSmiles(mol, isomericSmiles = True)) # Smiles

			# Check for redundancies and average
			if 'nr-' in dset or 'sr-' in dset or 'tox21' in dset:
				continue
				# Don't worry about duplicates in TOX21 dataset

			elif smiles.count(smiles[-1]) > 1:
				print('**** DUPLICATE ENTRY ****')
				print(smiles[-1])

				indices = [x for x in range(len(smiles)) if smiles[x] == smiles[-1]]
				y[indices[0]] = (y[indices[0]] + this_y) / 2.
				
				del y[-1]
				del smiles[-1]
				del mols[-1]


		except Exception as e:
			print('Failed to generate graph for {}, y: {}'.format(row[smiles_index], row[y_index]))
			print(e)

	###################################################################################
	### DIVIDE DATA VIA RATIO OR CV
	###################################################################################

	if 'ratio' in data_split: # split train/notrain
		print('Using first fraction ({}) as training'.format(training_ratio))
		# Create training/development split
		division = int(len(mols) * training_ratio)
		mols_train = mols[:division]
		mols_notrain  = mols[division:]
		y_train = y[:division]
		y_notrain  = y[division:]
		smiles_train = smiles[:division]
		smiles_notrain = smiles[division:]

		# Split notrain up
		mols_val    = mols_notrain[:(len(mols_notrain) / 2)] # first half
		y_val       = y_notrain[:(len(mols_notrain) / 2)] # first half
		smiles_val  = smiles_notrain[:(len(mols_notrain) / 2)] # first half
		mols_test   = mols_notrain[(len(mols_notrain) / 2):] # second half
		y_test      = y_notrain[(len(mols_notrain) / 2):] # second half
		smiles_test = smiles_notrain[(len(mols_notrain) / 2):] # second half
		print('Training size: {}'.format(len(mols_train)))
		print('Validation size: {}'.format(len(mols_val)))
		print('Testing size: {}'.format(len(mols_test)))

	elif 'all_train' in data_split: # put everything in train 
		print('Using ALL as training')
		# Create training/development split
		mols_train = mols
		y_train = y
		smiles_train = smiles
		mols_val    = []
		y_val       = []
		smiles_val  = []
		mols_test   = []
		y_test      = []
		smiles_test = []
		print('Training size: {}'.format(len(mols_train)))
		print('Validation size: {}'.format(len(mols_val)))
		print('Testing size: {}'.format(len(mols_test)))

	elif 'cv' in data_split: # cross-validation
		# Default to first fold of 5-fold cross-validation
		folds = 5
		this_fold = 0

		# Read fold information
		try:
			folds = int(cv_folds.split('/')[1])
			this_fold = int(cv_folds.split('/')[0]) - 1
		except:
			pass

		# Get target size of each fold
		N = len(mols)
		print('Total of {} mols'.format(N))
		target_fold_size = int(np.ceil(float(N) / folds))
		# Split up data
		folded_mols 	= [mols[x:x+target_fold_size]   for x in range(0, N, target_fold_size)]
		folded_y 		= [y[x:x+target_fold_size]      for x in range(0, N, target_fold_size)]
		folded_smiles 	= [smiles[x:x+target_fold_size] for x in range(0, N, target_fold_size)]
		print('Split data into {} folds'.format(folds))
		print('...using fold {}'.format(this_fold + 1))

		# Recombine into training and testing
		mols_train   = [x for fold in (folded_mols[:this_fold] + folded_mols[(this_fold + 1):])     for x in fold]
		y_train      = [x for fold in (folded_y[:this_fold] + folded_y[(this_fold + 1):])           for x in fold]
		smiles_train = [x for fold in (folded_smiles[:this_fold] + folded_smiles[(this_fold + 1):]) for x in fold]
		# Test is this_fold
		mols_test    = folded_mols[this_fold]
		y_test       = folded_y[this_fold]
		smiles_test  = folded_smiles[this_fold]

		# Define validation set as random 10% of training
		training_indices = range(len(mols_train))
		np.random.shuffle(training_indices)
		split = int(len(training_indices) * training_ratio)
		mols_train,   mols_val    = [mols_train[i] for i in training_indices[:split]],   [mols_train[i] for i in training_indices[split:]]
		y_train,      y_val       = [y_train[i] for i in training_indices[:split]],      [y_train[i] for i in training_indices[split:]]
		smiles_train, smiles_val  = [smiles_train[i] for i in training_indices[:split]], [smiles_train[i] for i in training_indices[split:]]

		print('Total training: {}'.format(len(mols_train)))
		print('Total validation: {}'.format(len(mols_val)))
		print('Total testing: {}'.format(len(mols_test)))

	else:
		print('Must specify a data_split type of "ratio" or "cv"')
		quit(1)

	


	###################################################################################
	### REPACKAGE AS DICTIONARIES
	###################################################################################
	if 'cv_full' in data_split: # cross-validation, but use 'test' as validation
		train = {}; train['mols'] = mols_train; train['y'] = y_train; train['smiles'] = smiles_train; train['y_label'] = y_label
		val   = {}; val['mols']   = mols_test;   val['y']   = y_test;   val['smiles']   = smiles_test;   val['y_label']   = y_label
		test  = {}; test['mols']  = [];  test['y']  = [];  test['smiles']  = []; test['y_label']  = []

	else:

		train = {}; train['mols'] = mols_train; train['y'] = y_train; train['smiles'] = smiles_train; train['y_label'] = y_label
		val   = {}; val['mols']   = mols_val;   val['y']   = y_val;   val['smiles']   = smiles_val;   val['y_label']   = y_label
		test  = {}; test['mols']  = mols_test;  test['y']  = y_test;  test['smiles']  = smiles_test; test['y_label']  = y_label

	return (train, val, test)
