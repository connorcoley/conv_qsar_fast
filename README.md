# conv_qsar_fast
QSAR/QSPR using descriptor-free molecular embedding

## Requirements
This code relies on [Keras](http://keras.io/) for the machine learning framework, [Theano](http://deeplearning.net/software/theano/) for computations as its back-end, and [RDKit](http://www.rdkit.org/) for parsing molecules from SMILES strings. Plotting is done in [matplotlib](http://matplotlib.org/). All other required packages should be dependencies of Keras, Theano, or RDKit.

## Basic use
This code implements the tensor-based convolutional embedding strategy described in __placeholder__ for QSAR/QSPR tasks. The model architecture, training schedule, and data source are defined in a configuration file and trained using a cross-validation (CV). The basic architecture is as follows:

- Pre-processing to convert a SMILES string into an attributed graph, then into an attributed adjacency tensor
- Convolutional embedding layer, which takings a molecular tensor and produces a learned feature vector
- Optional dropout layer
- Optional hidden densely-connected neural network layer
- Optional dropout layer
- Optional second hidden densely-connected neural network layer
- Linear dense output layer

Models are built, trained, and tested with the command
```
python conv_qsar_fast/main/main_cv.py conv_qsar_fast/inputs/<input_file>.cfg
```

Numerous example input files, corresponding the models described in __placeholder__ are included in `inputs`. These include models to be trained on full datasets, 5-fold CVs with internal validation and early stopping, 5-fold CVs without internal validation, models initialized with weights from other trained models, and multi-task models predicting on multiple data sets. Note that when using multi-task models, the `output_size` must be increased and the `loss` function must be `custom` to ensure `NaN` values are filtered out if not all inputs x have the full set of outputs y.

## Data sets
There are four available data sets in this version of the code contained in `data`:

1. Abraham octanol solubility data, from Abraham and Admire's 2014 paper.
2. Delaney aqueous solubility data, from Delaney's 2004 paper.
3. Bradley double plus good melting point data, from Bradley's open science notebook initiative.
4. Tox21 data from the Tox21 Data Challenge 2014, describing toxicity against 12 targets.

Because certain entries could not be unambiguously resolved into chemical structures, or because duplicates in the data sets were found, the effective data sets after processing are exported using `scripts/save_data.py` as `coley_abraham.tdf`, `coley_delaney.tdf`, `coley_bradley.tdf`, `coley_tox21.tdf`, and `coley_tox21-test.tdf`.

## Model interpretation
This version of the code contains the general method of non-linear model interpretation of assigning individual atom and bond attributes to their average value in the molecular tensor representation. The extent to which this hurts performance is indicative of how dependent a trained model has become on that atom/bond feature. As long as the configuration file defines a model which loads previously-trained weights, the testing routine is performed by
```
python conv_qsar_fast/main/test_index_removal.py conv_qsar_fast/inputs/<input_file>.cfg
```
It is assumed that the trained model used molecular_attributes, as the indices for removal are hard-coded into this script.

## Suggestions for modification
#### Overall architecture
The range of possible architectures (beyond what is enabled with the current configuration file style) can be extended by modifying `build_model` in `main/core.py`. See the Keras documentation for ideas.

#### Data sets
Additional `.csv` data sets can be incorporated by adding an additional `elif` statement to `main/data.py`. As long as one column corresponds to SMILES strings and another to the property target, the existing code can be used with minimal modification.

#### Atom-level or bond-level attributes
Additional atom- or bond-level attributes can be included by modifying `utils/neural_fp.py`, specifically the `bondAttributes` and `atomAttributes` functions. Because molecules are already stored as RDKit molecule objects, any property calculable in RDKit can easily be added.
