# train.py


from datasets import load_dataset
from transformers import (AutoTokenizer,
                          AutoModelForSequenceClassification,
                          TrainingArguments, 
                          Trainer)

import numpy as np
import evaluate
import logging
import ast

import os
from typing import Union
import re

# for updating an external results .csv file
import pandas as pd
from sklearn.metrics import confusion_matrix
 
from transformers import set_seed
from transformers import EarlyStoppingCallback, IntervalStrategy

def run_trainer(trainfile: str, 
                testfile: str, 
                trainfile_secondary: str,
                testfile_secondary: str,
                output_dir: str, 
                logger: Union[logging.Logger, None], 
                seed: int = 42) -> AutoModelForSequenceClassification:
    """Runs trainer

    Args:
        trainfile (str): Train file (.csv)
        testfile (str): Test file (.csv)
        output_dir (str): Output directory
        logger Union[logging.Logger, None], optional: Logger to use

    Returns:
        AutoModelForSequenceClassification: Trained model
    """
    
    # log = logging.getLogger(__name__) if logger is None else logger
    
    # sanity checks
    for i in [trainfile, testfile, output_dir]:
        assert os.path.exists(i), f"File/Directory {i} does not exist"
    
    # seed needs to be set before model instantiation for full reproducability of first run
    set_seed(42)
    
    # load model
    # log.info('loading model...')
    model = AutoModelForSequenceClassification.from_pretrained(os.getenv('MODEL_NAME'))
    
    # log.info('loading dataset...')
    dataset = load_dataset("csv", data_files={"train": trainfile, 
                                              "test": testfile})
    
    # define tokenizer and tokenize datasets
    # log.info('loading tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(os.getenv('TOKENIZER_NAME'), max_length=512)
    tokenizer.model_max_length = 512  

    # **************** ADDING SPECIAL TOKENS ****************** #
    special_tokens = ast.literal_eval(os.getenv('SPECIAL_TOKENS'))
    if (special_tokens != -1):
        special_tokens_dict = {'additional_special_tokens': special_tokens}
        tokenizer.add_special_tokens(special_tokens_dict)
        model.resize_token_embeddings(len(tokenizer)) 
    # ************************************************************* #
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, load_from_cache_file=False)
    
    # **************** FREEZING LAYERS ****************** #
    layers_to_freeze = ast.literal_eval(os.getenv('LAYERS_TO_FREEZE'))
    if (layers_to_freeze != -1):
        for name, param in model.named_parameters():
            if (re.search(r'\d+', name)): # if this layer has a number (if it doesn't, re.search() returns None)
                if (int(re.search(r'\d+', name).group(0)) in layers_to_freeze):
                    param.requires_grad = False # freezes the layer
        # output layer statuses to make sure
        for name, param in model.named_parameters(): # prints out layers and frozen status
            print(name, param.requires_grad)
    # ********************************************************* #

    # define training args
    training_args = TrainingArguments(output_dir = output_dir, 
                                      evaluation_strategy = "epoch", 
                                      num_train_epochs = float(os.getenv('NUM_EPOCHS')),
                                      learning_rate = float(os.getenv('LEARNING_RATE')),
                                      per_device_train_batch_size = int(os.getenv('BATCH_SIZE')),
                                      per_device_eval_batch_size = int(os.getenv('BATCH_SIZE')),
                                      warmup_steps = int(os.getenv('WARMUP_STEPS')),
                                      logging_strategy = 'epoch',
                                      logging_first_step = True,
                                      metric_for_best_model = 'f1',
                                      # lr_scheduler_type='constant',  # if want to keep learning rate fixed
                                      save_strategy = 'epoch',
                                      save_total_limit = 1,
                                      load_best_model_at_end = True,
                                     )
    
    # define evaluation metrics
    metric_f1 = evaluate.load("f1")
    metric_pr = evaluate.load("precision")
    metric_re = evaluate.load("recall")

    train_output = os.getenv('TRAIN_RESULTS_CSV')
    
    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        f1 = metric_f1.compute(predictions=predictions, 
                               references=labels, 
                               average='macro')
        
        recall = metric_re.compute(predictions=predictions, 
                                   references=labels,
                                  average='macro')
        
        precision = metric_pr.compute(predictions=predictions, 
                                      references=labels,
                                     average='macro')
        
        # ***** update an external results .csv file ***** # 
        lang = os.getenv('L1')
        
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        if (os.path.basename(train_output) not in os.listdir(os.getenv('EXP_DIR'))):
            df = pd.DataFrame(columns=['test_no', 'f1', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp', 'preds', 'lang'])
        else:
            df = pd.read_csv(train_output, index_col=0)
        file_no = re.search(r'\d+', os.path.basename(trainfile)).group(0)

        df.loc[len(df.index)] = [file_no, f1['f1'], precision['precision'], recall['recall'], tn, fp, fn, tp, predictions, lang]
        df.to_csv(train_output)
        
        return f1
    
 
    
    # define test and train splits
    train_dataset = tokenized_datasets["train"]
    test_dataset = tokenized_datasets["test"]
    
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE')))]
    )

    trainer.train()
    model.save_pretrained("./{}/fine-tuned-model".format(os.getenv('letter')))                                                       
    tokenizer.save_pretrained("./{}/fine-tuned-model".format(os.getenv('letter')))
    
    ############# SECONDARY TRAINING ###########################
    

    new_model = AutoModelForSequenceClassification.from_pretrained("./{}/fine-tuned-model".format(os.getenv('letter')))
    new_tokenizer = AutoTokenizer.from_pretrained("./{}/fine-tuned-model".format(os.getenv('letter')))

    
    # load the csv into a DatasetDict object, because it can utilize the map() function, used in the block below
    new_dataset = load_dataset("csv", data_files={"train": trainfile_secondary,
                                              "test": testfile_secondary})
    
    
    def tokenize_function(examples):
        return new_tokenizer(examples['text'], padding="max_length", truncation=True) # I don't understand the point of 'truncation'
    
    new_tokenized_dataset = new_dataset.map(tokenize_function, batched=True) # applies tokenize_function to each text

    new_train_dataset = new_tokenized_dataset['train']
    new_eval_dataset = new_tokenized_dataset['test']

    # adjust training settings below
    training_args = TrainingArguments(output_dir=output_dir,
                                      evaluation_strategy="epoch",
                                      num_train_epochs = float(os.getenv('NUM_EPOCHS')),
                                      learning_rate = float(os.getenv('LEARNING_RATE')),
                                      per_device_train_batch_size = int(os.getenv('BATCH_SIZE')),
                                      per_device_eval_batch_size = int(os.getenv('BATCH_SIZE')),
                                      logging_strategy = 'epoch',
                                      logging_first_step = True,
                                      # lr_scheduler_type='constant',  # if want to keep learning rate fixed
                                      save_strategy = 'epoch',
                                      load_best_model_at_end = True,
                                      metric_for_best_model = 'f1')

    import numpy as np
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    
    def compute_metrics(eval_pred):
        import numpy as np
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        
        f1 = metric_f1.compute(predictions=predictions, 
                               references=labels, 
                               average='macro')
        
        recall = metric_re.compute(predictions=predictions, 
                                   references=labels,
                                  average='macro')
        
        precision = metric_pr.compute(predictions=predictions, 
                                      references=labels,
                                     average='macro')
        
        # ***** update an external results .csv file ***** # 
        tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()
        lang = os.getenv('L2')
        
        if (os.path.basename(train_output) not in os.listdir(os.getenv('EXP_DIR'))):
            df = pd.DataFrame(columns=['test_no', 'f1', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp', 'preds', 'lang'])
        else:
            df = pd.read_csv(train_output, index_col=0)
        file_no = re.search(r'\d+', os.path.basename(trainfile_secondary)).group(0)

            
        df.loc[len(df.index)] = [file_no, f1['f1'], precision['precision'], recall['recall'], tn, fp, fn, tp, predictions, lang]
        df.to_csv(train_output)
        
        return f1
    
    
        # create a Trainer() object, which is used to initiate training
    trainer = Trainer(
        model=new_model,
        args=training_args,
        train_dataset=new_train_dataset,
        eval_dataset=new_eval_dataset,
        compute_metrics=compute_metrics,
        callbacks = [EarlyStoppingCallback(early_stopping_patience=int(os.getenv('EARLY_STOPPING_PATIENCE')))]
    )

    trainer.train() # actually start the training

    # Save the updated model
    new_model.save_pretrained("./{}/subsequently-tuned-model".format(os.getenv('letter')))
    new_tokenizer.save_pretrained("./{}/subsequently-tuned-model".format(os.getenv('letter')))

    
    # **************** POST-TRAINING EVALUATION PER LANGUAGE ****************** #
    # compute metrics 
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    from sklearn.metrics import accuracy_score

    # if evaluating, generate predictions using model and then return performance metrics
    if (bool(os.getenv('EVALUATION'))):
       
        test_output = os.getenv('TEST_RESULTS_CSV')       
        def use_model_for_euph_predictions(tokenizer, model, test_data):
            preds = [] 
            preds_DF = pd.DataFrame(test_data)
            # generate predictions    
            for i, row in test_data.iterrows():
                text = test_data.loc[i, 'text']
                inputs = tokenizer(text, return_tensors='pt', truncation=True).to('cuda')
                logits = model(**inputs).logits
                predicted_class_id = logits.argmax().item()
                preds.append(predicted_class_id)                
            # compute metrics
            preds_DF['predicted'] = preds
            labels = test_data['label'].tolist()
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average='macro')
            precision = precision_score(labels, preds, average='macro')
            recall = recall_score(labels, preds, average='macro')
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            
            return accuracy, f1, precision, recall, tn, fp, fn, tp, preds, preds_DF
    
        print("Evaluating best model on test sets...")
        # prepare to evaluate on each of the languages' test sets
        TEST_NUM = re.search(r'\d+', os.path.basename(trainfile)).group(0)
        LANGS = ast.literal_eval(os.getenv('LANGS')) # ['chinese', 'english', 'spanish', 'yoruba']
        results = []
        
        # output result to an external .csv file]
        if (os.path.basename(test_output) not in os.listdir(os.getenv('EXP_DIR'))):
            df = pd.DataFrame(columns=['TEST_NUM', 'LANG', 'accuracy', 'f1', 'precision', 'recall', 'tn', 'fp', 'fn', 'tp', 'preds'])
        else:
            df = pd.read_csv(test_output, index_col=0)
            
        
        # BULK TESTING MODIFIER
        bulk_modifier = int(os.getenv('BULK_TESTING_MODIFIER'))
        if (bulk_modifier != -1):
            TEST_NUM = str(int(TEST_NUM) % bulk_modifier)
        
        # evaluate for each of the languages' test sets
        for lang in LANGS:
            test_data = pd.read_csv("{}/test_{}_{}.csv".format(os.getenv('EXP_DIR'), TEST_NUM, lang), index_col=0) # language-specific test file; a CSV of test examples
            result = use_model_for_euph_predictions(new_tokenizer, new_model, test_data)
            df.loc[len(df.index)] = [TEST_NUM, lang, result[0], result[1], result[2], result[3], result[4], result[5], result[6], result[7], result[8]]
            df.to_csv(test_output)
            
    
    # save model
    if (os.getenv('SAVE_MODELS') == 'True'):
        trainer.save_model(output_dir)
    
    return model
