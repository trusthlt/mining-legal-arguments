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
from torch.utils.data.sampler import RandomSampler
from typing import List, Union, Dict
from transformers import DataCollatorForTokenClassification
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from typing import Optional, Any
import argparse
from tabulate import tabulate


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


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models. 

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name, 
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        elif model_class_name.startswith("DistilBert"):
            return "distilbert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


class StrIgnoreDevice(str):
    """
    This is a hack. The Trainer is going call .to(device) on every input
    value, but we need to pass in an additional `task_name` string.
    This prevents it from throwing an error
    """
    def to(self, device):
        return self


class DataLoaderWithTaskname:
    """
    Wrapper around a DataLoader to also yield a task name
    """
    def __init__(self, task_name, data_loader):
        self.task_name = task_name
        self.data_loader = data_loader

        self.batch_size = data_loader.batch_size
        self.dataset = data_loader.dataset

    def __len__(self):
        return len(self.data_loader)
    
    def __iter__(self):
        for batch in self.data_loader:
            batch["task_name"] = StrIgnoreDevice(self.task_name)
            yield batch


class MultitaskDataloader:
    """
    Data loader that combines and samples from multiple single-task
    data loaders.
    """
    def __init__(self, dataloader_dict):
        self.dataloader_dict = dataloader_dict
        self.num_batches_dict = {
            task_name: len(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        self.task_name_list = list(self.dataloader_dict)
        self.dataset = [None] * sum(
            len(dataloader.dataset) 
            for dataloader in self.dataloader_dict.values()
        )

    def __len__(self):
        return sum(self.num_batches_dict.values())

    def __iter__(self):
        """
        For each batch, sample a task, and yield a batch from the respective
        task Dataloader.

        We use size-proportional sampling, but you could easily modify this
        to sample from some-other distribution.
        """
        task_choice_list = []
        for i, task_name in enumerate(self.task_name_list):
            task_choice_list += [i] * self.num_batches_dict[task_name]
        task_choice_list = np.array(task_choice_list)
        np.random.shuffle(task_choice_list)
        dataloader_iter_dict = {
            task_name: iter(dataloader) 
            for task_name, dataloader in self.dataloader_dict.items()
        }
        for task_choice in task_choice_list:
            task_name = self.task_name_list[task_choice]
            yield next(dataloader_iter_dict[task_name])    


class MultitaskTrainer(transformers.Trainer):

    def get_single_train_dataloader(self, task_name, train_dataset):
        """
        Create a single-task data loader that also yields task names
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if is_torch_tpu_available():
            train_sampler = get_tpu_sampler(train_dataset)
        else:
            train_sampler = (
                RandomSampler(train_dataset)
                if self.args.local_rank == -1
                else DistributedSampler(train_dataset)
            )

        data_loader = DataLoaderWithTaskname(
            task_name=task_name,
            data_loader=DataLoader(
              train_dataset,
              batch_size=self.args.train_batch_size,
              sampler=train_sampler,
              collate_fn=self.data_collator.collate_batch,
            ),
        )

        if is_torch_tpu_available():
            data_loader = pl.ParallelLoader(
                data_loader, [self.args.device]
            ).per_device_loader(self.args.device)
        return data_loader


    def get_train_dataloader(self):
        """
        Returns a MultitaskDataloader, which is not actually a Dataloader
        but an iterable that returns a generator that samples from each 
        task Dataloader.
        """
        return MultitaskDataloader({
            task_name: self.get_single_train_dataloader(task_name, task_dataset)
            for task_name, task_dataset in self.train_dataset.items()
        })

    def get_eval_dataloader(self, q):
        """
        Returns a DataLoaderWithTaskname for the argument type task
        for evaluation of it during the training.
        """
        eval_dataloader_argType = DataLoaderWithTaskname(
        'ArgType',
        data_loader=DataLoader(
              eval_dataset['ArgType'],
              batch_size=trainer.args.eval_batch_size,
              collate_fn=trainer.data_collator.collate_batch,
            ),
        )
        return eval_dataloader_argType

    def save_model(self, output_dir: Optional[str] = None):
        """
        Saving best-practices: if you use default names for the model,
        you can reload it using from_pretrained().
        Will only save from the world_master process (unless in TPUs).
        """
        if is_torch_tpu_available():
            self._save_tpu(output_dir)
        elif self.is_world_process_zero():
            self._save(output_dir)

    def _save_tpu(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info("Saving model checkpoint to %s", output_dir)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))



        xm.rendezvous("saving_checkpoint")
        torch.save(self.model, os.path.join(output_dir, self.model.encoder.base_model_prefix))

    def _save(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)
        # Low-Level workaround for MultiTaskModel
        torch.save(self.model, os.path.join(output_dir, self.model.encoder.base_model_prefix))

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


@dataclasses.dataclass
class MyDataCollatorForTokenClassification:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    
    # call not used?
    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if labels is None:
            return batch
    
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]

        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


    def collate_batch(self, features, pad_to_multiple_of: Optional[int] = None):
        
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )
        
        if labels is None:
            return batch
        
        del batch['tokens']
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
        batch = {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()}
        return batch


def compute_f1(label, gold, pred):
    """
    Computes the F1 Score for a single class.
    :param labal: the class to compute the score for
    :param gold: the gold standard
    :param pred: the model predictions
    :return: the F1 score for the label"""
    tp = 0
    fp = 0
    fn = 0
    
    for i, sent in enumerate(pred):
        for j, tag in enumerate(sent):
            # check for relevant label to compute F1
            if tag == label:
                # if relevant and equals gold -> true positive
                if tag == gold[i][j]:
                    tp += 1
                # if it differs from gold -> false positive
                else: 
                    fp += 1
            # we have a negative, so check if it's a false negative
            else:
                if gold[i][j] == label:
                    fn += 1
    # use epsilon to avoid division by zero
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1

def compute_macro_f1(gold, pred, id2label):
    """
    Computes the Macro F1 Score over all classes.
    :param gold: the gold standard
    :param pred: the model predictions
    :param id2label: the mapping list for the current labels
    :return: the Macro F1 score"""
    f1s = [(tag, compute_f1(tag, gold, pred)) for tag in range(len(id2label))]
    
    all_f1s = [(id2label[idx], score) for idx, score in f1s]

    df = pd.DataFrame(all_f1s, columns=['Label', 'F1'])
    df['F1'] = np.around(df['F1'], decimals=4)
    print(tabulate(df, headers='keys', tablefmt='pretty', showindex=False))
    
    f1_scores = [f1[1] for f1 in f1s]
    macro_f1 = np.sum(f1_scores) / len(f1_scores)
    #print('Macro F1: ', macro_f1)
    return macro_f1

def eval_f1(evalpred):
    """
    Computes the Macro F1 Score over all argument type classes during train evaluation.
    :param evalpred: evalpred from the trainer
    :return: the Macro F1 score"""
    pred = []
    gold = []
    for p,l in zip(np.argmax(evalpred.predictions, axis=2), evalpred.label_ids):
        ind = np.logical_and(p > -1, l > -1)
        pred.append(p[ind])
        gold.append(l[ind])
    
    f1s = [(tag, compute_f1(tag, gold, pred)) for tag in range(len(id2label_argType))]
    
    all_f1s = [(id2label_argType[idx], score) for idx, score in f1s]
    #print('F1 for each Class: ', all_f1s)
    
    f1_scores = [f1[1] for f1 in f1s]
    macro_f1 = np.sum(f1_scores) / len(f1_scores)
    return {"F1 ArgType": macro_f1}


if __name__ == '__main__':

    # parse optional args
    parser = argparse.ArgumentParser(description='Train a MultiTask model')
    parser.add_argument('--pathprefix', help='path to the project directory')
    parser.add_argument('--model', help='name of the model or path to the model')
    parser.add_argument('--tokenizer', help='name of the model or path to the tokenizer')
    parser.add_argument('--batch_size', type=int, help='batch size of the model')
    parser.add_argument('--output_dir', help='path to the output directory')

    args = parser.parse_args()

    # path to working directory
    pathprefix = '/ukp-storage-1/dfaber/'
    #pathprefix = ''
    if args.pathprefix:
        pathprefix = args.pathprefix

    # load datasets
    trainfiles = [f for f in os.listdir(pathprefix + 'data/train/argType/') if f.endswith('.csv')]
    valfiles = [f for f in os.listdir(pathprefix + 'data/val/argType/') if f.endswith('.csv')]


    dataset_argType = load_dataset('csv', data_files={'train': [pathprefix + 'data/train/argType/' + file for file in trainfiles],
                                                  'validation': [pathprefix + 'data/val/argType/' + file for file in valfiles]}, delimiter='\t')

    dataset_actor = load_dataset('csv', data_files={'train': [pathprefix + 'data/train/agent/' + file for file in trainfiles],
                                                  'validation': [pathprefix + 'data/val/agent/' + file for file in valfiles]}, delimiter='\t')

    dataset_argType = dataset_argType.map(lambda x: {'tokens': literal_eval(x['tokens']), 'labels': literal_eval(x['labels'])})
    dataset_actor = dataset_actor.map(lambda x: {'tokens': literal_eval(x['tokens']), 'labels': literal_eval(x['labels'])})


    # select the model with the correspronding tokenizer

    #model_name = "/ukp-storage-1/dfaber/models/court_bert/checkpoint-20000"
    #tokenizer = AutoTokenizer.from_pretrained('/ukp-storage-1/dfaber/legal_tokenizer_bert', do_lower_case=False)

    model_name = "/ukp-storage-1/dfaber/models/roberta-large-finetuned/checkpoint-15000"
    #model_name = 'roberta-large'
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')

    #model_name = 'nlpaueb/legal-bert-base-uncased'
    #tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-base-uncased')

    # use parsed args if provided
    if args.model:
        model_name = args.model
    if args.tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # need prefix space for already tokenized data
    if 'roberta' in model_name:
        tokenizer.add_prefix_space = True
    if tokenizer.model_max_length > 1024:
        tokenizer.model_max_length = 512

    # tokenize and align labels
    tokenized_dataset_argType = dataset_argType.map(tokenize_and_align_labels_argType, batched=True)
    tokenized_dataset_actor = dataset_actor.map(tokenize_and_align_labels_agent, batched=True)


    # create multitask dataset  
    dataset_dict = {
        "ArgType": tokenized_dataset_argType,
        "Actor": tokenized_dataset_actor,
    }


    # create multitask model
    multitask_model = MultitaskModel.create(
        model_name=model_name,
        model_type_dict={
            "ArgType": transformers.AutoModelForTokenClassification,
            "Actor": transformers.AutoModelForTokenClassification,
        },
        model_config_dict={
            "ArgType": transformers.AutoConfig.from_pretrained(model_name, num_labels=len(id2label_argType)),
            "Actor": transformers.AutoConfig.from_pretrained(model_name, num_labels=len(id2label_agent)),
        },
    )


    # create data collator
    data_collator= MyDataCollatorForTokenClassification(tokenizer)

    # split dataset into training and evaluation (dev) dataset
    train_dataset = {
        task_name: dataset["train"] 
        for task_name, dataset in dataset_dict.items()
    }
    eval_dataset = {
        task_name: dataset["validation"] 
        for task_name, dataset in dataset_dict.items()
    }

    # set training parameter and train the model
    output_dir = pathprefix + 'models/multitask/roberta-large-fp-15000'
    batch_size = 4
    # use parsed if provided
    if args.output_dir:
        output_dir = args.output_dir
    if args.batch_size:
        batch_size = args.batch_size

    train_args = transformers.TrainingArguments(
        output_dir,
        evaluation_strategy = "epoch",
        logging_steps=1592,
        learning_rate=1e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=10,
        weight_decay=0.01,
        warmup_steps=1000,
        save_steps=15926,
        save_total_limit = 10,
        logging_dir=pathprefix + 'logs',
    )
    trainer = MultitaskTrainer(
        model=multitask_model,
        args=train_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        compute_metrics=eval_f1,
    )
    trainer.train()


