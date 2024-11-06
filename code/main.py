## Load Libraries
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import evaluate
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from datetime import datetime

from sklearn.model_selection import train_test_split

from dataloader import BERTDataset
from constants import *


if torch.cuda.is_available():
            DEVICE = torch.device('cuda')
elif torch.mps.is_available():
    DEVICE = torch.device('mps')


class Model:
    def __init__(self, model_name=MODEL_NAME):
        # ## Set Hyperparameters
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

        
        # ## Load Tokenizer and Model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=7).to(DEVICE)
 
        # ## Define Dataset
        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
        
        # ## Define Metrics
        self.metric = evaluate.load(METRICS)
        
        print(self.model, self.tokenizer, self.metric)
    
    def load_train_dataset(self, dataset:pd.DataFrame):
        dataset_train, dataset_valid = train_test_split(dataset, test_size=TEST_SPLIT_SIZE, random_state=SEED)
        
        self.data_train = BERTDataset(dataset_train, self.tokenizer)
        self.data_valid = BERTDataset(dataset_valid, self.tokenizer)
        
    def compute_metrics(self,eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return self.metric.compute(predictions=predictions, references=labels, average='macro')
    
    def train(self, dataset:pd.DataFrame):
        # Train Model
        
        ### for wandb setting
        os.environ['WANDB_DISABLED'] = 'true'

        
        self.load_train_dataset(dataset)
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            overwrite_output_dir=True,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy='steps',
            eval_strategy='steps',
            save_strategy='steps',
            logging_steps=100,
            eval_steps=100,
            save_steps=100,
            save_total_limit=2,
            learning_rate= 2e-05,
            adam_beta1 = 0.9,
            adam_beta2 = 0.999,
            adam_epsilon=1e-08,
            weight_decay=0.01,
            lr_scheduler_type='linear',
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=2,
            load_best_model_at_end=True,
            metric_for_best_model='eval_f1',
            greater_is_better=True,
            seed=SEED
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data_train,
            eval_dataset=self.data_valid,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()        
        
    def evaluate(self, dataset_test:pd.DataFrame):
        # ## Evaluate Model
        self.model.eval()
        preds = []

        for idx, sample in tqdm(dataset_test.iterrows(), total=len(dataset_test), desc="Evaluating"):
            inputs = self.tokenizer(sample['text'], return_tensors="pt").to(DEVICE)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred = torch.argmax(torch.nn.Softmax(dim=1)(logits), dim=1).cpu().numpy()
                preds.extend(pred)

        # saving outputs
        dataset_test['target'] = preds
        
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"output_basic_{current_time}.csv"

        dataset_test.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)
        print(filename + 'has been saved successfully!')

