import rdkit.Chem as Chem
import numpy as np
import os


'''
This script is meant to split the Tox21 train dataset into the 
individual target datasets for training single-task models.
'''


if __name__ == '__main__':

	# Read SDF
	suppl = Chem.SDMolSupplier(
		os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'tox21_10k_data_all.sdf'
		),
		sanitize = False
	)

	mols = []
	smiles = []
	ys = None
	targets = [
		'NR-AhR',
		'NR-AR',
		'NR-AR-LBD',
		'NR-Aromatase',
		'NR-ER',
		'NR-ER-LBD',
		'NR-PPAR-gamma',
		'SR-ARE',
		'SR-ATAD5',
		'SR-HSE',
		'SR-MMP',
		'SR-p53'
	]
	j = 1
	for mol in suppl:
		mols.append(mol)
		smiles.append(Chem.MolToSmiles(mol))
		y = np.nan * np.ones((1, len(targets)))
		for i, target in enumerate(targets):
			try:
				y[0, i] = bool(float(mol.GetProp(target)))
			except Exception as e:
				pass
		if type(ys) == type(None): 
			ys = y
		else:
			ys = np.concatenate((ys, y))
		if j % 500 == 0:
			print('completed {} entries'.format(j))
		j += 1
	
	print(ys)
	print(ys.shape)
	for i, target in enumerate(targets):
		print('Target {} has {} entries; {} active'.format(
			target, sum(~np.isnan(ys[:, i])), np.sum(ys[~np.isnan(ys[:, i]), i])
		))
		
	with open(os.path.join(
			os.path.dirname(os.path.dirname(__file__)),
			'data', 'tox21.smiles'
			), 'w') as fid:
		for j, smile in enumerate(smiles):
			fid.write('{}\t{}\t{}\n'.format(smile, '??', '\t'.join([str(x) for x in ys[j, :]])))

