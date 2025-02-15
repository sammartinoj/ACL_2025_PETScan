#launch py
import os
import logging
import time
from datetime import datetime
from train import run_trainer

####################################################################
''' (0) SET EXPERIMENTAL DIRECTORY '''
####################################################################
EXP_DIR = "../test_data" # filepath to directory containing train/val/test splits; outputs 
####################################################################
''' (1) CONFIGURE EXPERIMENT SETTINGS (AS ENVIRONMENT VARIABLES) '''
####################################################################

# CHANGE
# ranges: chinese 0-9 english 20-29 spanish 40-49 yoruba 60-69 turkish 80-89
os.environ['START_FILENUM'] = "0"
os.environ['END_FILENUM'] = "10" # non-inclusive 
os.environ['TRAIN_RESULTS_CSV'] = f'{EXP_DIR}/results_train_ch_yo.csv'
os.environ['TEST_RESULTS_CSV'] = f'{EXP_DIR}/results_test_ch_yo.csv'
os.environ['letter'] = 'a' # used to separate checkpoints if running multiple combinations at once 
os.environ['LANGS'] =  "['chinese', 'yoruba']" 
os.environ['DIFF'] = '+60' # see note in readme file for different 'difference' values
os.environ['L1'] = 'chinese'
os.environ['L2'] = 'yoruba'


# model names
os.environ['MODEL_NAME'] = "xlm-roberta-base" # 'dccuchile/bert-base-spanish-wwm-cased' #  'xlm-roberta-large' , 'bert-base-multilingual-cased'
os.environ['TOKENIZER_NAME'] = "xlm-roberta-base" # 'dccuchile/bert-base-spanish-wwm-cased' #  'xlm-roberta-large' , 'bert-base-multilingual-cased'
os.environ['EXP_DIR'] = EXP_DIR # leave this as is
os.environ['SAVE_MODELS'] = 'True' # if True, will output best models for each trial into a folder called "saved_models" in EXP_DIR

# personal note created in EXP_DIR
os.environ['LOG'] = 'testing testing 1 2 3'

# some of the arguments in TrainingArguments that are commonly adjusted for fine-tuning in `train.py`
os.environ['NUM_EPOCHS'] = '15' 
os.environ['LEARNING_RATE'] = '1e-5' 
os.environ['WARMUP_STEPS'] = '0' # default: 0
os.environ['BATCH_SIZE'] = '4' 

# training settings
os.environ['EARLY_STOPPING_PATIENCE'] = "5" # a callback variable for custom Trainer behavior controlling Patience
os.environ['LAYERS_TO_FREEZE'] = '-1' 
os.environ['SPECIAL_TOKENS'] = '-1'


# evaluation settings
os.environ['EVALUATION'] = 'True' # whether to run evaluation at all
os.environ['BULK_TESTING_MODIFIER'] = "5"


####################################################################
''' (2) RUN FINE-TUNING AND EVALUATION ON EACH EXPERIMENT FILE '''
####################################################################

from manifest import manifest # import the manifest now (now that setup is complete)

logging.basicConfig(format='[%(name)s] [%(levelname)s] %(asctime)s %(message)s')

# ---------- log meta-info ---------- #
f = open(f"{EXP_DIR}/notes.txt", "a")
f.write(str(datetime.now()) + '\n')
f.write(os.getenv('LOG') + '\n')

for name, value in os.environ.items():
    f.write(f"{name}: {value}\n")
f.close()

start = time.time()
# ----------------------------------- #

# iterate through each item in the manifest
for item in manifest:
    # define train and test directories
    traindir = os.path.join(EXP_DIR, item['trainfile'])
    testdir = os.path.join(EXP_DIR, item['testfile'])
    traintwodir = os.path.join(EXP_DIR, item['trainfile_secondary'])
    testtwodir = os.path.join(EXP_DIR, item['testfile_secondary'])
    
    model_output_dir = os.path.join(EXP_DIR, 'saved_models_{}'.format(os.getenv('letter')), item['model_name'])
    # make model output dir
    os.system(f'mkdir -p {model_output_dir}')
    
    preds_folder = os.path.join(EXP_DIR, 'predictions')
    os.system(f'mkdir -p {preds_folder}')
    
    # make sure train and test files exist
    if not all(os.path.exists(i) for i in [traindir, testdir, traintwodir, testtwodir, model_output_dir]):
        logging.critical('missing dataset file(s) for model %s', item['model_name'])
    else:
        # define logger
        # Create and configure logger
        logger = logging.getLogger(item['model_name'])
        logger.setLevel(logging.INFO)

        # run trainer
        model = run_trainer(traindir, 
                            testdir, 
                            traintwodir,
                            testtwodir,
                            model_output_dir, 
                            logger)
    del model

# ---------- log meta-info ---------- #
duration = time.time() - start
f = open(f"{EXP_DIR}/notes.txt", "a")
f.write("DURATION: " + str(duration) + '\n\n')
f.close()
# ----------------------------------- #