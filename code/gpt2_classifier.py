import argparse
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Tokenizer
from transformers import BloomTokenizerFast, BloomForSequenceClassification
from sklearn.metrics import classification_report
from transformers import pipeline

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
    
    tokenizer = GPT2Tokenizer.from_pretrained("./results/GPT2_Plantl_Classification")
    
    model = AutoModelForSequenceClassification.from_pretrained("./results/GPT2_Plantl_Classification", num_labels=len(label2id)).to(device)
    
    classifier = pipeline("text-classification", model = model, tokenizer=tokenizer, device = 0)
    
    
    result_df = pd.DataFrame()
    
    prediction_dict = classifier(test_df['tweet_clean'].values.tolist())
    prediction = [item['label'] for item in prediction_dict]
    result_df['pred'] = prediction
    result_df['label'] = test_df['label'].values
    
    print(classification_report(result_df['label'], result_df['pred'], digits = 5))
    result_df.to_csv("./results/gpt2_plantl_results.csv", index=False)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/dataset.csv')
    parser.add_argument('--save_path', type=str, default='./results')
    args = parser.parse_args()
    main(args)