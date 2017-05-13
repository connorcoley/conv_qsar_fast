from __future__ import print_function
import rdkit.Chem as Chem
from conv_qsar_fast.utils.neural_fp import molToGraph
import conv_qsar_fast.utils.stats as stats
import keras.backend as K 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

def test_model(model, data, fpath, tstamp = 'no_time', batch_size = 128, return_test_MSE = False, verbose = False):
	'''This function evaluates model performance using test data.

	inputs:
		model - the trained Keras model
		data - three dictionaries for training,
					validation, and testing data. Each dictionary should have
					keys of 'mol', a molecular tensor, 'y', the target output, 
					and 'smiles', the SMILES string of that molecule
		fpath - folderpath to save test data to, will be appended with '/tstamp.test'
		tstamp - timestamp to add to the testing
		batch_size - batch_size to use while testing'''
	
	# Create folder to dump testing info to
	try:
		os.makedirs(fpath)
	except: # file exists
		pass
	test_fpath = os.path.join(fpath, tstamp)

	# Unpack
	(train, val, test) = data
	# Unpack
	mols_train = train['mols']; y_train = train['y']; smiles_train = train['smiles']
	mols_val   = val['mols'];   y_val   = val['y'];   smiles_val   = val['smiles']
	mols_test  = test['mols'];  y_test  = test['y'];  smiles_test  = test['smiles']

	y_train_pred = []
	y_val_pred = []
	y_test_pred = []

	if batch_size == 1: # UNEVEN TENSORS, ONE AT A TIME PREDICTION
		# Run through training set
		for j in tqdm(range(len(mols_train))):
			single_mol = mols_train[j]
			spred = model.predict_on_batch([np.array([single_mol[0]]), np.array([single_mol[1]]), np.array([single_mol[2]])])		
			y_train_pred.append(spred)

		# Run through validation set
		for j in tqdm(range(len(mols_val))):
			single_mol = mols_val[j]
			spred = model.predict_on_batch([np.array([single_mol[0]]), np.array([single_mol[1]]), np.array([single_mol[2]])])	
			y_val_pred.append(spred)

		# Run through testing set
		for j in tqdm(range(len(mols_test))):
			single_mol = mols_test[j]
			spred = model.predict_on_batch([np.array([single_mol[0]]), np.array([single_mol[1]]), np.array([single_mol[2]])])	
			y_test_pred.append(spred)

	else: # PADDED
		y_train_pred = np.array([]); y_val_pred = np.array([]); y_test_pred = np.array([])
		if mols_train: y_train_pred = model.predict(np.array(mols_train), batch_size = batch_size, verbose = 1)
		if mols_val: y_val_pred = model.predict(np.array(mols_val), batch_size = batch_size, verbose = 1)
		if mols_test: y_test_pred = model.predict(np.array(mols_test), batch_size = batch_size, verbose = 1)

	def round3(x):
		return int(x * 1000) / 1000.0

	def parity_plot(true, pred, set_label):
		if len(true) == 0:
			print('skipping parity plot for empty dataset')
			return

		try:
			# Trim it to recorded values (not NaN)
			true = np.array(true).flatten()
			if verbose: print(true)
			if verbose: print(true.shape)
			pred = np.array(pred).flatten()
			if verbose: print(pred)
			if verbose: print(pred.shape)

			pred = pred[~np.isnan(true)]
			true = true[~np.isnan(true)]

			print('{}:'.format(set_label))

			# For TOX21
			AUC = 'N/A'
			if len(set(list(true))) <= 2:
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
				plt.savefig(test_fpath + ' {} ROC.png'.format(set_label), bbox_inches = 'tight')
				plt.clf()
				print('  AUC = {}'.format(AUC))

			min_y = np.min((true, pred))
			max_y = np.max((true, pred))
			mse = stats.mse(true, pred)
			mae = stats.mae(true, pred)
			q = stats.q(true, pred)
			(r2, a) = stats.linreg(true, pred) # predicted v observed
			(r2p, ap) = stats.linreg(pred, true) # observed v predicted

			# Print
			print('  mse = {}, mae = {}'.format(mse, mae))
			if verbose:
				print('  q = {}'.format(q))
				print('  r2 through origin = {} (pred v. true), {} (true v. pred)'.format(r2, r2p))
				print('  slope through origin = {} (pred v. true), {} (true v. pred)'.format(a[0], ap[0]))

			# Create parity plot
			plt.scatter(true, pred, alpha = 0.5)
			plt.xlabel('Actual')
			plt.ylabel('Predicted')
			plt.title('Parity plot for {} ({} set, N = {})'.format(y_label, set_label, len(true)) + 
				'\nMSE = {}, MAE = {}, q = {}, AUC = {}'.format(round3(mse), round3(mae), round3(q), AUC) + 
				'\na = {}, r^2 = {}'.format(round3(a), round3(r2)) + 
				'\na` = {}, r^2` = {}'.format(round3(ap), round3(r2p)))
			plt.grid(True)
			plt.plot(true, true * a, 'r--')
			plt.axis([min_y, max_y, min_y, max_y])	
			plt.savefig(test_fpath + ' {}.png'.format(set_label), bbox_inches = 'tight')
			plt.clf()

			if len(set(list(true))) <= 2:
				return AUC
			return mse

		except Exception as e:
			print(e)
			return 99999

	# Create plots for datasets
	if y_train:
		y_label = train['y_label']
		if type(y_train[0]) != type(0.0):
			num_targets = y_train[0].shape[-1]
		else:
			num_targets = 1
	elif y_val:
		y_label = val['y_label']
		if type(y_val[0]) != type(0.0):
			num_targets = y_val[0].shape[-1]
		else:
			num_targets = 1
	elif y_test:
		y_label = test['y_label']
		if type(y_test[0]) != type(0.0):
			num_targets = y_test[0].shape[-1]
		else:
			num_targets = 1
	else:
		raise ValueError('Nothing to evaluate?')

	# Save
	with open(test_fpath + '.test', 'w') as fid:
		fid.write('{} tested {}, predicting {}\n\n'.format(fpath, tstamp, y_label))		
		fid.write('test entry\tsmiles\tactual\tpredicted\tactual - predicted\n')
		for i in range(len(smiles_test)):
			fid.write('{}\t{}\t{}\t{}\t{}\n'.format(i, 
				smiles_test[i],
				y_test[i], 
				y_test_pred[i],
				y_test[i] - y_test_pred[i]))

	test_MSE = 99999
	if y_train: 
		if type(y_train[0]) != type(0.): 
			num_targets = len(y_train[0])
			print('Number of targets: {}'.format(num_targets))
			for i in range(num_targets):
				parity_plot([x[i] for x in y_train], [x[0, i] for x in y_train_pred], 'train - ' + y_label[i])
		else:
			parity_plot(y_train, y_train_pred, 'train')
	if y_val: 
		if type(y_val[0]) != type(0.): 
			num_targets = len(y_val[0])
			print('Number of targets: {}'.format(num_targets))
			for i in range(num_targets):
				parity_plot([x[i] for x in y_val], [x[0, i] for x in y_val_pred], 'val - ' + y_label[i])
		else:
			parity_plot(y_val, y_val_pred, 'test')
	if y_test: 
		if type(y_test[0]) != type(0.): 
			num_targets = len(y_test[0])
			print('Number of targets: {}'.format(num_targets))
			test_MSE = 0.
			for i in range(num_targets):
				test_MSE += parity_plot([x[i] for x in y_test], [x[0, i] for x in y_test_pred], 'test - ' + y_label[i])
		else:
			test_MSE = parity_plot(y_test, y_test_pred, 'test')

	# train['residuals'] = np.array(y_train) - np.array(y_train_pred)
	# val['residuals'] = np.array(y_val) - np.array(y_val_pred)
	# test['residuals'] = np.array(y_test) - np.array(y_test_pred)

	if return_test_MSE: return test_MSE

	return (train, val, test)

def test_embeddings_demo(model, fpath):
	'''This function tests molecular representations by creating visualizations
	of fingerprints given a SMILES string. Molecular attributes are used, so the
	model to load should have been trained using molecular attributes.

	inputs:
		model - the trained Keras model
		fpath - folderpath to save test data to, will be appended with '/embeddings/'
	'''
	print('Building images of fingerprint examples')

	# Create folder to dump testing info to
	try:
		os.makedirs(fpath)
	except: # folder exists
		pass
	try:
		fpath = os.path.join(fpath, 'embeddings')
		os.makedirs(fpath)
	except: # folder exists
		pass

	# Define function to test embedding
	x = K.placeholder(ndim = 4)
	tf = K.function([x], 
		model.layers[0].call(x))

	# Define function to save image
	def embedding_to_png(embedding, label, fpath):
		print(embedding)
		print(embedding.shape)
		fig = plt.figure(figsize=(20,0.5))
		plt.pcolor(embedding, vmin = 0, vmax = 1, cmap = plt.get_cmap('Greens'))
		plt.title('{}'.format(label))
		# cbar = plt.colorbar()
		plt.gca().yaxis.set_visible(False)
		plt.gca().xaxis.set_visible(False)
		plt.xlim([0, embedding.shape[1]])
		plt.subplots_adjust(left = 0, right = 1, top = 0.4, bottom = 0)
		plt.savefig(os.path.join(fpath, label) + '.png', bbox_inches = 'tight')
		with open(os.path.join(fpath, label) + '.txt', 'w') as fid:
			fid.write(str(embedding))
		plt.close(fig)
		plt.clf()
		return

	smiles = ''
	print('**using molecular attributes**')
	while True:
		smiles = raw_input('Enter smiles: ').strip()
		if smiles == 'done':
			break
		try:
			mol = Chem.MolFromSmiles(smiles)
			mol_graph = molToGraph(mol, molecular_attributes = True).dump_as_tensor()
			single_mol_as_array = np.array([mol_graph])
			embedding = tf([single_mol_as_array])
			with open(os.path.join(fpath, smiles) + '.embedding', 'w') as fid:
				for num in embedding.flatten():
					fid.write(str(num) + '\n')
			embedding_to_png(embedding, smiles, fpath)
		except:
			print('error saving embedding - was that a SMILES string?')

	return
