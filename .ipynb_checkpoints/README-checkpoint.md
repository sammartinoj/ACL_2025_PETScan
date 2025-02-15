# ACL_2025_PETScan

Instructions for running PETScan. 

1. Prepare an experiment folder consisting of files called "train_X.csv", "val_X.csv", and "test_X_LANG.csv". If testing on multiple languages, ensure that each train/val file has their own range of X values, but the testing files all have the same numbers. i.e., if testing on chinese and english, your files will look like:
    - train_0.csv, train_1.csv... train_4.csv... val_0.csv, val_1.csv...val_4.csv, test_0_chinese.csv, test_1_chinese.csv, ....
    - train_20.csv, train_21.csv... train_24.csv... val_20.csv, val_21.csv...val_24.csv, test_0_english.csv, test_1_english.csv, ...
    
    For our data, the ranges used for file names is:
    0-4 - chinese
    20-24 - english
    40-44 - spanish
    60-64 - yoruba
    80-84 - turkish

2. Make sure local/run.sh is correctly configured, and if this is the first time using the experiment runner, ensure that the 'requirements' line is uncommented. 

3. Ensure src/launch.py values are correctly set - including the exp_dir path. **IMPORTANT** - the lines labelled 'NEED CHANGED' are to instantiate what language pair you wish to experiment with. 

     ** For 'DIFF' environment variable:
         if L1 is chinese,  +20 - english, +40 - spanish, +60 - yoruba, +80 - turkish
         if L1 is english,  -20 - chinese, +20 - spanish,  +40 - yoruba,  +60 - turkish
         if L1 is spanish,  -40 - chinese, -20 - english, +20 - yoruba, +40 - turkish
         if L1 is yoruba,   -60 - chinese, -40 - english, -20 - spanish,  +20 - turkish
         if L1 is turkish,  -80 - chinese, -60 - english, -40 - spanish, -20 - yoruba
                       
4. From a terminal, 'cd' into the PETScan folder. 

5. Enable permissions to execute the script ('chmod +x local/run.sh')

6. launch training with sh local/run.sh