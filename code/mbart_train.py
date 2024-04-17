import argparse
import torch
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import transformers
import evaluate
from transformers import set_seed
from transformers import TrainingArguments
from transformers import Trainer
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import AutoModelForSequenceClassification
from transformers import BertModel
from transformers import RobertaModel
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report
import torch.nn as nn
from transformers import BartTokenizer, BartForSequenceClassification
from sklearn.metrics import accuracy_score
from datasets import Dataset, load_metric
from transformers import BartForSequenceClassification, Trainer, TrainingArguments, EvalPrediction

# reproductilibty
set_seed(1)
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)
# GPU device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_path = "facebook/mbart-large-50"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast = True)


metric1 = evaluate.load("precision")
metric2 = evaluate.load("recall")
metric3 = evaluate.load("f1")
metric_name = "f_macro"


# def compute_metrics(p: EvalPrediction):
#   metric_acc = load_metric("accuracy")
#   metric_f1 = load_metric("f1")
#   preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
#   preds = np.argmax(preds, axis = 1)
#   result = {}
#   result["accuracy"] = metric_acc.compute(predictions = preds, references = p.label_ids)["accuracy"]
#   result["f1"] = metric_f1.compute(predictions = preds, references = p.label_ids, average = 'macro')["f1"]
#   print(classification_report(p.label_ids, preds, digits=5))
#   return result

def compute_metrics(pred):
    # logits, labels = eval_pred
    # predictions = np.argmax(logits, axis=-1)
    labels_ids = pred.label_ids
    pred_ids = pred.predictions[0]
    
    print(classification_report(labels_ids, pred_ids, digits=4))
    precision = metric1.compute(predictions=pred_ids, references=labels_ids, average='weighted')["precision"]
    recall = metric2.compute(predictions=pred_ids, references=labels_ids, average='weighted')["recall"]
    f_score = metric3.compute(predictions=pred_ids, references=labels_ids, average='weighted')["f1"]
    f_macro = metric3.compute(predictions=pred_ids, references=labels_ids, average='macro')["f1"]
    return {"precision": precision, "recall": recall, "f_score": f_score, "f_macro": f_macro}
    

def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels
  

def preprocess_function(examples):
  result = tokenizer(examples["tweet_clean"])
  # result['input_ids_sentiment'] = tokenized_inputs['input_ids']
  # result['attention_mask_sentiment'] = tokenized_inputs['attention_mask']
  result['labels'] = examples['label']
  return result
  
 
def train_model(model, train_dataset, eval_dataset, save_path):

    # Use the best hyperparameter after the search
    batch_train_size = 1
    batch_eval_size = 1
    EPOCHS = 8
    training_args = TrainingArguments(
        output_dir = './results',
        # overwrite_output_dir=True,
        num_train_epochs=EPOCHS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit = 1,
        per_device_train_batch_size=batch_train_size,
        per_device_eval_batch_size=batch_eval_size,
        metric_for_best_model=metric_name,
        fp16=True,
        weight_decay=0.01,
        learning_rate=2e-5,
        warmup_steps = 500,    
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        tokenizer = tokenizer
    )
    
    trainer.train()
    # Salvamos el modelo reentrenado
    trainer.save_model(f'{save_path}/mbart_classification')


def main(args): 
  
    dataset_path = args.dataset_path
    save_path = args.save_path

    dataset = pd.read_csv(dataset_path)
    dataset = dataset[["tweet_clean", "__split", "label"]].dropna()
    print(dataset)
    label_list = dataset.label.unique().tolist()
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for i, label in enumerate(label_list)}
    print(label_list)
    print(label2id)
    print(id2label)
        
    dataset['label'] = dataset['label'].apply(lambda x: label2id[x])
    #dataset['sentiment_logits'] = dataset['tweet_clean'].apply(lambda x: get_logits(x))
    # dataset['text_features'] = dataset['tweet_clean'].apply(lambda x: list(text_features(x).values()))
    # print(dataset.info())
    
    # Split the dataset 
    train_df = dataset[dataset['__split']=='train']
    val_df = dataset[dataset['__split']=='val']
    test_df = dataset[dataset['__split']=='test']
    
    # Load the model and tokenizers 
    num_label = len(dataset.label.unique())
    
    # Parser the dataframe to dataset dict
    train_df = Dataset.from_pandas(train_df)
    val_df = Dataset.from_pandas(val_df)
    test_df = Dataset.from_pandas(test_df)
    
    train_dataset = train_df.map(
        preprocess_function,
        batched=True,
        num_proc=4
    )

    eval_dataset = val_df.map(
        preprocess_function,
        batched=True,
        num_proc=4
    )

    test_dataset = test_df.map(
        preprocess_function,
        batched=True,
        num_proc=4
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=len(label2id), trust_remote_code=True)
    
    train_model(model, train_dataset, eval_dataset, save_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/dataset.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)