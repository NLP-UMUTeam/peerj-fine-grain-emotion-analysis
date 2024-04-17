import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from sklearn.metrics import classification_report
from transformers import pipeline
from peft import PeftModel    


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
        
    # dataset['label'] = dataset['label'].apply(lambda x: label2id[x])
    
    # Split the dataset 
    train_df = dataset[dataset['__split']=='train']
    val_df = dataset[dataset['__split']=='val']
    test_df = dataset[dataset['__split']=='test']
    
    model_path = "bigscience/bloom-3b"
    adapter_model = "./results/BloomLoraClassification"
    tokenizer = BloomTokenizerFast.from_pretrained(model_path)
    
    config = AutoConfig.from_pretrained(model_path, label2id=label2id, id2label=id2label)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config).to(device)
    model = PeftModel.from_pretrained(model, adapter_model)
    model = model.merge_and_unload()
    
    classifier = pipeline("text-classification", model = model, tokenizer=tokenizer, device = 0)
    
    result_df = pd.DataFrame()
    
    prediction_dict = classifier(test_df['tweet_clean'].values.tolist())
    prediction = [item['label'] for item in prediction_dict]
    result_df['pred'] = prediction
    result_df['label'] = test_df['label'].values
    
    print(classification_report(result_df['label'], result_df['pred'], digits = 5))
    result_df.to_csv(f"{save_path}/bloom_lora_results.csv", index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/dataset.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)