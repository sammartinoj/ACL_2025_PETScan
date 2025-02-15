# manifest.py
import os

manifest = []
start_filenum = os.getenv('START_FILENUM')
end_filenum = os.getenv('END_FILENUM')
exp_dir = os.getenv('EXP_DIR')
difference = int(os.getenv('DIFF'))

for x in range(int(start_filenum), int(end_filenum)):
    num = str(x)
    num_2 = str(x+difference)
    test_obj = {'model_name': 'finetuned_'+num,
     'trainfile': 'train_{}.csv'.format(num),
     'testfile': 'val_{}.csv'.format(num),
     'trainfile_secondary': 'train_{}.csv'.format(num_2),
     'testfile_secondary': 'val_{}.csv'.format(num_2)}
    manifest.append(test_obj)