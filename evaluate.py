#!/usr/bin/env python
# coding: utf-8


from collections import Counter
from prettytable import PrettyTable
import os 
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset, load_metric
import csv
from ast import literal_eval
import numpy as np
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, load_metric
import logging
import dataclasses
from torch.utils.data.dataloader import DataLoader
from transformers.training_args import is_torch_tpu_available
from transformers.trainer_pt_utils import get_tpu_sampler
from transformers.data.data_collator import DataCollator, InputDataClass
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from typing import List, Union, Dict
from transformers import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from typing import Optional, Any
from sklearn.metrics import confusion_matrix
from multiTaskModel import MultitaskModel, StrIgnoreDevice, DataLoaderWithTaskname, MultitaskDataloader, MultitaskTrainer, MyDataCollatorForTokenClassification, compute_f1, compute_macro_f1, eval_f1
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


id2label_argType = ['B-Distinguishing',
 'B-Einschätzungsspielraum',
 'B-Entscheidung des EGMR',
 'B-Konsens der prozessualen Parteien',
 'B-Overruling',
 'B-Rechtsvergleichung',
 'B-Sinn & Zweck Auslegung',
 'B-Subsumtion',
 'B-Systematische Auslegung',
 'B-Verhältnismäßigkeitsprüfung – Angemessenheit',
 'B-Verhältnismäßigkeitsprüfung – Geeignetheit',
 'B-Verhältnismäßigkeitsprüfung – Legitimer Zweck',
 'B-Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
 'B-Vorherige Rechtsprechung des EGMR',
 'B-Wortlaut Auslegung',
 'I-Distinguishing',
 'I-Einschätzungsspielraum',
 'I-Entscheidung des EGMR',
 'I-Konsens der prozessualen Parteien',
 'I-Overruling',
 'I-Rechtsvergleichung',
 'I-Sinn & Zweck Auslegung',
 'I-Subsumtion',
 'I-Systematische Auslegung',
 'I-Verhältnismäßigkeitsprüfung – Angemessenheit',
 'I-Verhältnismäßigkeitsprüfung – Geeignetheit',
 'I-Verhältnismäßigkeitsprüfung – Legitimer Zweck',
 'I-Verhältnismäßigkeitsprüfung – Rechtsgrundlage',
 'I-Vorherige Rechtsprechung des EGMR',
 'I-Wortlaut Auslegung',
 'O']
label2id_argType = {}
for i, label in enumerate(id2label_argType):
    label2id_argType[label] = i
    
id2label_agent = ['B-Beschwerdeführer',
 'B-Dritte',
 'B-EGMR',
 'B-Kommission/Kammer',
 'B-Staat',
 'I-Beschwerdeführer',
 'I-Dritte',
 'I-EGMR',
 'I-Kommission/Kammer',
 'I-Staat',
 'O']
label2id_agent = {}
for i, label in enumerate(id2label_agent):
    label2id_agent[label] = i
    

def tokenize_and_align_labels_argType(examples, label_all_tokens=False):
    """
    Tokenizes the input using the tokenizer and aligns the argument type labels to the subwords.
    :param examples: input dataset
    :param label_all_tokens: Whether to label all subwords of a token or only the first subword
    :return: Tokenized input"""
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id_argType[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label2id_argType[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
    
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def tokenize_and_align_labels_agent(examples, label_all_tokens=False):
    """
    Tokenizes the input using the tokenizer and aligns the agent labels to the subwords.
    :param examples: input dataset
    :param label_all_tokens: Whether to label all subwords of a token or only the first subword
    :return: Tokenized input"""
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i) 
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id_agent[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(label2id_agent[label[word_idx]] if label_all_tokens else -100)
            previous_word_idx = word_idx
    
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs



def get_subset_df(tokens, predictions, labels, predlabel=None, truelabel=None):
    """
    Can filter model predictions by predicted and true label. If none provided, just postprocesses and returns output.
    :param tokens: text tokens
    :param predictions: predictions of the model
    :param labels: true annotator labels
    :param predlabel: optional, filters the model predictions if a label is provided, e.g. label2id_agent['I-EGMR']
    :param truelabels: optional, filters the true annotator labels if a label is provided, e.g. label2id_agent['I-EGMR']
    :return: DataFrame with the (filtered) predictions, labels and tokens."""
    pred = []
    label = []
    for p,l in zip(predictions, labels):
        p = np.array(p)
        l = np.array(l)
        ind = np.logical_and(p > -1, l > -1)
        pred.append(p[ind].tolist())
        label.append(l[ind].tolist())

    if predlabel is None and truelabel is None:
        return pd.DataFrame({'Predictions': pred, 'Labels': label, 'Tokens': tokens})
    elif predlabel is None:
        preds = []
        labels = []
        toks = []
        for i,l in enumerate(label):
            if truelabel in l[2:] and not truelabel in pred[i][2:]:
                preds.append(pred[i])
                labels.append(l)
                toks.append(tokens[i])

        return pd.DataFrame({'Predictions': preds, 'Labels': labels, 'Tokens': toks})
    elif truelabel is None:
        preds = []
        labels = []
        toks = []
        for i,p in enumerate(pred):
            if predlabel in p[2:] and not predlabel in label[i][2:]:
                preds.append(p)
                labels.append(llabel[i])
                toks.append(tokens[i])

        return pd.DataFrame({'Predictions': preds, 'Labels': labels, 'Tokens': toks})
    else:
        preds = []
        labels = []
        toks = []
        for i,p in enumerate(pred):
            if predlabel in p[2:] and truelabel in label[i][2:]:
                preds.append(p)
                labels.append(label[i])
                toks.append(tokens[i])

        return pd.DataFrame({'Predictions': preds, 'Labels': labels, 'Tokens': toks})


def save_predictions(tokens, predictions, labels, file):
    """
    Saves the model predictions as csv after postprocessing them.
    :param tokens: text tokens
    :param predictions: predictions of the model
    :param labels: true annotator labels
    :param file: path of the output file
    """
    df = get_subset_df(tokens, predictions, labels)
    df.to_csv(file , sep='\t', encoding='utf-8', index=False)


if __name__ == '__main__':

    # parse optional args
    parser = argparse.ArgumentParser(description='Evaluate a MultiTask model and save its predictions')
    parser.add_argument('--pathprefix', help='path to the project directory')
    parser.add_argument('--models', nargs='*' ,help='paths to the models to evaluate')
    parser.add_argument('--test_dir', help='path to the directory with the test files')
    parser.add_argument('--val_dir', help='path to the directory with the dev files')
    parser.add_argument('--output_dir', help='path to the output directory for saving the predictions')
    parser.add_argument('--do_val', default=False, type=lambda x: (str(x).lower() == 'true'),  help='whether to evaluate the validation/dev dataset')
    parser.add_argument('--do_test', default=True, type=lambda x: (str(x).lower() == 'true'),  help='whether to evaluate the test dataset')

    args = parser.parse_args()
    
    # project directory
    pathprefix = '/ukp-storage-1/dfaber/'
    pathprefix = '../Uni/masterthesis/'
    pathprefix = ''
    if args.pathprefix:
        pathprefix = args.pathprefix
    
    #test_dir = 'data/article_3/'
    test_dir = 'data/test/'
    if args.test_dir:
        test_dir = args.test_dir

    val_dir = 'data/val/'
    if args.val_dir:
        val_dir = args.val_dir

    output_dir = 'predictions/'
    if args.output_dir:
        output_dir = args.output_dir

    # load datasets
    testfiles = [f for f in os.listdir(os.path.join(pathprefix, test_dir, 'argType/')) if f.endswith('.csv')]
    valfiles = [f for f in os.listdir(os.path.join(pathprefix, val_dir, 'argType/')) if f.endswith('.csv')]
    testfiles = testfiles[:2]
    valfiles = valfiles[:2]


    dataset_argType = load_dataset('csv', data_files={'test': [os.path.join(pathprefix, test_dir, 'argType/', file) for file in testfiles],
                                                  'validation': [os.path.join(pathprefix, val_dir, 'argType/', file) for file in valfiles]}, delimiter='\t')

    dataset_actor = load_dataset('csv', data_files={'test': [os.path.join(pathprefix, test_dir, 'agent/', file) for file in testfiles],
                                                  'validation': [os.path.join(pathprefix, val_dir, 'agent/', file) for file in valfiles]}, delimiter='\t')

    dataset_argType = dataset_argType.map(lambda x: {'tokens': literal_eval(x['tokens']), 'labels': literal_eval(x['labels'])})
    dataset_actor = dataset_actor.map(lambda x: {'tokens': literal_eval(x['tokens']), 'labels': literal_eval(x['labels'])})

    # models to evaluate
    '''
    models = ['/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-39820/bert', '/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-47784/bert', 
              '/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-55748/bert', '/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-71676/bert', 
              '/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-79640/bert', '/ukp-storage-1/dfaber/models/multitask/roberta-large-final/checkpoint-111482/roberta',
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-final/checkpoint-143334/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-final/checkpoint-159260/roberta',
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-13000/checkpoint-95556/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-13000/checkpoint-127408/roberta',
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-13000/checkpoint-143334/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-13000/checkpoint-159260/roberta',
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-111482/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-127408/roberta',
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-143334/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-159260/roberta',
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-final/checkpoint-143334/roberta']
    '''

    models = ['/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-39820/bert', '/ukp-storage-1/dfaber/models/multitask/legal-bert-final/checkpoint-47784/bert', 
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-final/checkpoint-111482/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-final/checkpoint-143334/roberta', 
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-13000/checkpoint-95556/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-13000/checkpoint-143334/roberta', 
              '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-143334/roberta', '/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-159260/roberta']

    models = ['/ukp-storage-1/dfaber/models/multitask/roberta-large-fp-15000/checkpoint-143334/roberta']
    if args.models:
        models = args.models

    # Evaluate each model
    for model in models:
        print('\n\n\n\n********************Evaluating ', model, '********************\n\n\n\n')

        # load model and tokenizer
        multitask_model = torch.load(model) 
    
        tokenizer = AutoTokenizer.from_pretrained(multitask_model.encoder.name_or_path)
    
        if model.split('/')[-1] == 'roberta':
            tokenizer.add_prefix_space = True
        if tokenizer.model_max_length > 1024:
            tokenizer.model_max_length = 512
        
        # preprocess data and create datasets
        tokenized_dataset_argType = dataset_argType.map(tokenize_and_align_labels_argType, batched=True)
        tokenized_dataset_actor = dataset_actor.map(tokenize_and_align_labels_agent, batched=True)
    
        dataset_dict = {
        "ArgType": tokenized_dataset_argType,
        "Actor": tokenized_dataset_actor,
        }
    
        data_collator= MyDataCollatorForTokenClassification(tokenizer)
    
        test_dataset = {
            task_name: dataset["test"] 
            for task_name, dataset in dataset_dict.items()
        }
        val_dataset = {
            task_name: dataset["validation"] 
            for task_name, dataset in dataset_dict.items()
        }
        
        # initialize Trainer
        batch_size = 8
        train_args = transformers.TrainingArguments(
            'test_bert/legal_bert/',
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
        )
    
        trainer = MultitaskTrainer(
            model=multitask_model,
            args=train_args,
            data_collator=data_collator,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=eval_f1,
        )
        
        # evaluate validation data if specified
        if args.do_val:
            print('\n\n*****VALIDATION DATASET*****\n\n')

            eval_dataloader_argType = DataLoaderWithTaskname(
                'ArgType',
                data_loader=DataLoader(
                    val_dataset['ArgType'],
                    batch_size=trainer.args.eval_batch_size,
                    collate_fn=trainer.data_collator.collate_batch,
                    ),
            )
            preds_arg = trainer.prediction_loop(eval_dataloader_argType, description='Validation ArgType')
    
            eval_dataloader_agent = DataLoaderWithTaskname(
                    'Actor',
                    data_loader=DataLoader(
                        val_dataset['Actor'],
                        batch_size=trainer.args.eval_batch_size,
                        collate_fn=trainer.data_collator.collate_batch,
                        ),
                )
            preds_agent = trainer.prediction_loop(eval_dataloader_agent, description='Validation Agent')
    
            # postprocess (remove -100 indices)
            labels_argType_wordlevel = []
            preds_argType_wordlevel = []
            for l,p in zip(preds_arg.label_ids, np.argmax(preds_arg.predictions, axis=2)):
                ind = np.logical_and(p > -1, l > -1)
                labels_argType_wordlevel.append(l[ind])
                preds_argType_wordlevel.append(p[ind])
        
            print('ArgType:')
            print('Macro F1: ', compute_macro_f1(gold=labels_argType_wordlevel, pred=preds_argType_wordlevel, id2label=id2label_argType))
            
            # postprocess (remove -100 indices)
            labels_agent_wordlevel = []
            preds_agent_wordlevel = []
            for l,p in zip(preds_agent.label_ids, np.argmax(preds_agent.predictions, axis=2)):
                ind = np.logical_and(p > -1, l > -1)
                labels_agent_wordlevel.append(l[ind])
                preds_agent_wordlevel.append(p[ind])
        
            print('Agent:')
            print('Macro F1: ', compute_macro_f1(gold=labels_agent_wordlevel, pred=preds_agent_wordlevel, id2label=id2label_agent))
            
            # save predictions
            save_predictions(val_dataset['ArgType']['tokens'], np.argmax(preds_arg.predictions, axis=2), preds_arg.label_ids, os.path.join(pathprefix, output_dir, 'val_preds/', '_'.join(model.split('/')[-3:]) + '-argType.csv'))
            save_predictions(val_dataset['Actor']['tokens'], np.argmax(preds_agent.predictions, axis=2), preds_agent.label_ids, os.path.join(pathprefix, output_dir, 'val_preds/', '_'.join(model.split('/')[-3:]) + '-agent.csv'))


        # evaluate test data if specified
        if args.do_test:
            print('\n\n*****TEST DATASET*****\n\n')

            eval_dataloader_argType = DataLoaderWithTaskname(
                'ArgType',
                data_loader=DataLoader(
                    test_dataset['ArgType'],
                    batch_size=trainer.args.eval_batch_size,
                    collate_fn=trainer.data_collator.collate_batch,
                    ),
            )
            preds_arg = trainer.prediction_loop(eval_dataloader_argType, description='Validation ArgType')
    
            eval_dataloader_agent = DataLoaderWithTaskname(
                    'Actor',
                    data_loader=DataLoader(
                        test_dataset['Actor'],
                        batch_size=trainer.args.eval_batch_size,
                        collate_fn=trainer.data_collator.collate_batch,
                        ),
                )
            preds_agent = trainer.prediction_loop(eval_dataloader_agent, description='Validation Agent')
    
            # postprocess (remove -100 indices)
            labels_argType_wordlevel = []
            preds_argType_wordlevel = []
            for l,p in zip(preds_arg.label_ids, np.argmax(preds_arg.predictions, axis=2)):
                ind = np.logical_and(p > -1, l > -1)
                labels_argType_wordlevel.append(l[ind])
                preds_argType_wordlevel.append(p[ind])
        
            print('ArgType:')
            print('Macro F1: ', compute_macro_f1(gold=labels_argType_wordlevel, pred=preds_argType_wordlevel, id2label=id2label_argType))
            
            # postprocess (remove -100 indices)
            labels_agent_wordlevel = []
            preds_agent_wordlevel = []
            for l,p in zip(preds_agent.label_ids, np.argmax(preds_agent.predictions, axis=2)):
                ind = np.logical_and(p > -1, l > -1)
                labels_agent_wordlevel.append(l[ind])
                preds_agent_wordlevel.append(p[ind])
        
            print('Agent:')
            print('Macro F1: ', compute_macro_f1(gold=labels_agent_wordlevel, pred=preds_agent_wordlevel, id2label=id2label_agent))
            
            # save predictions
            save_predictions(test_dataset['ArgType']['tokens'], np.argmax(preds_arg.predictions, axis=2), preds_arg.label_ids, os.path.join(pathprefix, output_dir, '_'.join(model.split('/')[-3:]) + '-argType.csv'))
            save_predictions(test_dataset['Actor']['tokens'], np.argmax(preds_agent.predictions, axis=2), preds_agent.label_ids, os.path.join(pathprefix, output_dir, '_'.join(model.split('/')[-3:]) + '-agent.csv'))





