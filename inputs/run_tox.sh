
### SINGLE-TASK TOX21 MODELS 

# Train ST models on each task individually
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_ppar-gamma.cfg

# Test ST models by pooling the 5 models from the 5 CV folds
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ar.cfg   
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_er.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_mmp.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_p53.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_are.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_hse.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_ppar-gamma.cfg

# Train MORGAN FP RADIUS 3 (512)
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan/tox21_ppar-gamma.cfg

# Test MORGAN FP RADIUS 3 (512)
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_er.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_mmp.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ar.cfg   
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_p53.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_are.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_hse.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan/tox21_ppar-gamma.cfg

# Train MORGAN FP RADIUS 2 (512)
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ahr.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ar-lbd.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_er.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_mmp.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ar.cfg   
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_aromatase.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_er-lbd.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_p53.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_are.cfg  
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_atad5.cfg      
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_hse.cfg     
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21_Morgan2/tox21_ppar-gamma.cfg

# Test MORGAN FP RADIUS 2 (512)
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ahr.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ar-lbd.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_er.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_mmp.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ar.cfg   
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_aromatase.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_er-lbd.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_p53.cfg
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_are.cfg  
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_atad5.cfg      
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_hse.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21_Morgan2/tox21_ppar-gamma.cfg

### MULTITASK TOX21 MODELS 

python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_all_halfsize.cfg
python conv_qsar/main/main_cv.py conv_qsar/inputs/tox21/tox21_all_fullsize.cfg

python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_all_halfsize.cfg     
python conv_qsar/scripts/consensus_test_from_CV.py conv_qsar/inputs/tox21/tox21_all_fullsize.cfg