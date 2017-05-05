from __future__ import print_function
import os
from conv_qsar.main.data import get_data_full


'''
This script is meant to load the five datasets (abraham, delaney, 
bradley, tox21, and tox21-test) from their original sources and dump
them to thier own files. 

'''

if __name__ == '__main__':
	
	# Get *FULL* dataset
	data_kwargs = {}
	data_kwargs['batch_size'] = 1
	data_kwargs['training_ratio'] = 1.0
	data_kwargs['cv_folds'] = '1/1'
	data_kwargs['shuffle_seed'] = 0

	##############################
	### DEFINE TESTING CONDITIONS
	##############################

	datasets = [
		'abraham',
		'delaney',
		'bradley',
		'tox21',
		'tox21-test',
	]

	for j, dataset in enumerate(datasets):
		print('DATASET: {}'.format(dataset))

		data_kwargs['data_label'] = dataset
		data = get_data_full(**data_kwargs)[2] # testing set contains it all

		# Save
		with open(os.path.join(os.path.dirname(__file__), 'coley_' + dataset + '.tdf'), 'w') as fid:
			
			if type(data['y_label']) != type([]):
				fid.write('SMILES\t' + data['y_label'] + '\n')
			else:
				fid.write('SMILES\t{}\n'.format('\t'.join(data['y_label'])))

			for j in range(len(data['y'])): # for each mol in that list
				if type(data['y'][j]) != type([]):
					data_write = data['y'][j]  
				else:
					data_write = '\t'.join(data['y'][j])
				fid.write('{}\t{}\n'.format(data['smiles'][j], data_write))
