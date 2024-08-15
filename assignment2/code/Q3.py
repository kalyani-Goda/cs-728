import torch
from DTW import *
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from dataset import *
from datasets import load_dataset, load_metric
from functools import partial
import time
import os
import json
import pickle
import argparse

def getBack(var_grad_fn):
    print(var_grad_fn)
    for n in var_grad_fn.next_functions:
        if n[0]:
            try:
                tensor = getattr(n[0], 'variable')
                print(n[0])
                print('Tensor with grad found:', tensor)
                print(' - gradient:', tensor.grad)
                print()
            except AttributeError as e:
                getBack(n[0])


def sep_collate_fn(data, tokenizer):
    input_sents = [i[0] for i in data]
    sent1_batch = [sent1 for sent1, _ in input_sents]
    sent2_batch = [sent2 for _, sent2 in input_sents]
    labels = [i[1] for i in data]
    sent1_encoded = tokenizer(sent1_batch, padding="max_length", max_length = 32, truncation=True, return_tensors='pt')
    sent2_encoded = tokenizer(sent2_batch, padding="max_length", max_length = 32, truncation=True, return_tensors='pt')
    return sent1_encoded, sent2_encoded, labels

class TDWModelTrainer:
    """
    A class that can help in training the bert-tiny for the STS task using DTW sim score.
    reduce the learing in half after each epoch.
    """
    def __init__(self,pre_trained_model_name, device, crossing = False):
        self.task = "stsb"
        self.device = device
        self.model = DTW(pre_trained_model_name, crossing = crossing).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=0.005)
        self.metric = load_metric('glue', self.task)
        return

    def train(self, train_loader, val_loader, EPOCH):
        """
        Method to train the model.
        """
        loss_history = {"train_loss":[], "val_loss":[]}
        
        for epoch in range(EPOCH):
            print("Epoch :", epoch)
            if epoch and epoch%5 == 0:
                print("reducing learing rate by a factor of 0.7")
                for g in self.optimizer.param_groups:
                    g['lr'] *= 0.7
            self.model.train()
            train_loss = 0
            for sent1_encoded, sent2_encoded, labels in tqdm(train_loader):
                self.model.train()
                input_ids_1 = torch.tensor(sent1_encoded['input_ids']).to(self.device)
                input_attention_mask_1 = torch.tensor(sent1_encoded['attention_mask']).to(self.device)
                input_token_type_ids_1 = torch.tensor(sent1_encoded['token_type_ids']).to(self.device)
                input_ids_2 = torch.tensor(sent2_encoded['input_ids']).to(self.device)
                input_attention_mask_2 = torch.tensor(sent2_encoded['attention_mask']).to(self.device)
                input_token_type_ids_2 = torch.tensor(sent2_encoded['token_type_ids']).to(self.device)
                labels_t = (torch.tensor(labels).to(self.device)*2) / 5 - 1
                self.optimizer.zero_grad()
                outputs = self.model(input_ids_1 = input_ids_1,
                                          attention_mask_1 = input_attention_mask_1,
                                            token_type_ids_1 = input_token_type_ids_1,
                                            input_ids_2 = input_ids_2,
                                            attention_mask_2 = input_attention_mask_2,
                                            token_type_ids_2 = input_token_type_ids_2 
                                        )
                loss = torch.nn.functional.mse_loss(outputs.view(-1, 1), labels_t.view(-1, 1))
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            train_loss /= len(train_loader)
            print("Train Loss:", train_loss)
            loss_history["train_loss"].append(train_loss)
            val_loss, (targets, preds) = self.eval(val_loader, save_file = "output/Q3_1_val.txt")
            loss_history["val_loss"].append(val_loss)
            
        return
    
    def eval(self, val_loader, save_file=None):
        """
        Method to perform evaluation on the model.
        """
        self.model.eval()
        val_loss = 0
        max_lim = 1000
        cnt = 0
        with torch.no_grad():
            predictions = []
            targets = []
            for sent1_encoded, sent2_encoded, labels in tqdm(val_loader):
                input_ids_1 = torch.tensor(sent1_encoded['input_ids']).to(self.device)
                input_attention_mask_1 = torch.tensor(sent1_encoded['attention_mask']).to(self.device)
                input_token_type_ids_1 = torch.tensor(sent1_encoded['token_type_ids']).to(self.device)
                input_ids_2 = torch.tensor(sent2_encoded['input_ids']).to(self.device)
                input_attention_mask_2 = torch.tensor(sent2_encoded['attention_mask']).to(self.device)
                input_token_type_ids_2 = torch.tensor(sent2_encoded['token_type_ids']).to(self.device)
                labels_t = (torch.tensor(labels).to(self.device)*2) / 5 - 1
                outputs = self.model(input_ids_1 = input_ids_1,
                                          attention_mask_1 = input_attention_mask_1,
                                            token_type_ids_1 = input_token_type_ids_1,
                                            input_ids_2 = input_ids_2,
                                            attention_mask_2 = input_attention_mask_2,
                                            token_type_ids_2 = input_token_type_ids_2 
                                        )
                loss = torch.nn.functional.mse_loss(outputs.view(-1, 1), labels_t.view(-1, 1))
                val_loss += loss.item()
                outputs = outputs.squeeze()
                predictions.extend(outputs.tolist())
                targets.extend(labels_t)
                cnt += 1
                if cnt == max_lim:
                    break
            val_loss /= len(val_loader)
            print("Val Loss:", val_loss)
            print("Sim Metric:", self.metric.compute(predictions=predictions, references=targets))
            if save_file:
                print("Saving outputs in ", save_file)
                with open(save_file,'w') as ofile:
                    for item, _ in enumerate(predictions):
                        ofile.write(str(predictions[item])+str(targets[item])+"\n")

        return val_loss, (targets, predictions)

    def save_model(self, model_dir = None):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        file_name = "DTW"
        if self.model.crossing:
            file_name += "_crossing"
        else:
            file_name += "_non_crossing"
        file_name = model_dir + file_name + ".pt"

        torch.save({"a": self.model.a,"b": self.model.b},
                file_name)
        return
    
    def load_model(self, model_dir = None):
        file_name = "DTW"
        if self.model.crossing:
            file_name += "_crossing"
        else:
            file_name += "_non_crossing"

        file_name = model_dir + file_name + ".pt"

        print("Loading a,b from ", file_name)
        try:
            params = torch.load(file_name)
            self.model.a = params["a"]
            self.model.b = params["b"]
            print("a,b: ",self.model.a, self.model.b)
        except:
            print("Failed to load model from ", file_name)
        return 

    def test_sentence(self, crossing):
        """
        Method to perform inference on the sentences.
        """
        print("Press Ctrl+C to end this prompt:")
        while True:
            sent1 = str(input("Enter Sentence 1: "))
            sent2 = str(input("Enter Sentence 2: "))
            print(sent1)
            print(sent2)
            encoded_sent1 = self.tokenizer(sent1, return_tensors='pt')
            encoded_sent2 = self.tokenizer(sent2, return_tensors='pt')
            self.model.eval()
            with torch.no_grad():
                output1, _ = self.model.bert_model(input_ids = encoded_sent1['input_ids'], attention_mask = encoded_sent1['attention_mask'], token_type_ids = encoded_sent1['token_type_ids'])
                output2, _ = self.model.bert_model(input_ids = encoded_sent2['input_ids'], attention_mask = encoded_sent2['attention_mask'], token_type_ids = encoded_sent2['token_type_ids'])
                #print(output1)
                s1_emb = torch.squeeze(output1)[1:-1]
                s2_emb = torch.squeeze(output2)[1:-1]
                sim_score, k = self.model.get_DTW_score(s1_emb, s2_emb, return_map = True, crossing = crossing)
                tokenized_sent1 = self.tokenizer.convert_ids_to_tokens(encoded_sent1['input_ids'][0])[1:-1]
                tokenized_sent2 = self.tokenizer.convert_ids_to_tokens(encoded_sent2['input_ids'][0])[1:-1]
                if len(tokenized_sent1) < len(tokenized_sent2):
                    tokenized_sent1, tokenized_sent2 = tokenized_sent2, tokenized_sent1
                print(tokenized_sent1)
                print(tokenized_sent2)
                print("Sim Score:", (sim_score[0] + 1)*5/2)
                print("Allignment:", k)
                for i,_k in enumerate(list(k)):
                    print(tokenized_sent2[i], "\t:  ", tokenized_sent1[_k] if _k != None and _k != -1 else None)
    
def main():
    parser = argparse.ArgumentParser(description="STS using BERT-Tiny",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c", "--crossing", action='store_true', help="Whether to allow crossing between allignment.")
    parser.add_argument("-t", "--train", action='store_true', help="Whether to train the model")
    parser.add_argument("-i", "--infer", action='store_true', help="Whether to infer using command line")
    parser.add_argument("-e", "--epoch", default=1, help="number of epoch to train")
    parser.add_argument("-b", "--batchsz", default=16, help="Size of the batch to use while training")
    args = vars(parser.parse_args())
    
    BATCH_SIZE = int(args["batchsz"])
    EPOCH = int(args["epoch"])
    train = args["train"]
    infer = args["infer"]
    crossing = args["crossing"]

    task = "stsb"
    pre_trained_model_name = "prajjwal1/bert-tiny"
    seed = 42
    if crossing:
        c_name = "crossing"
    else:
        c_name = "non_crossing"

    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ",device)

    torch.manual_seed(seed)

    trainer = TDWModelTrainer(pre_trained_model_name, device, crossing = crossing)
    
    if train:
        dataset = load_dataset("glue", task)
        train_dataset = STSDataset(dataset["train"])
        val_dataset = STSDataset(dataset["validation"])
        test_dataset = STSDataset(dataset["test"])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn =partial(sep_collate_fn, tokenizer = trainer.tokenizer))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,collate_fn = partial(sep_collate_fn, tokenizer = trainer.tokenizer))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,collate_fn = partial(sep_collate_fn, tokenizer = trainer.tokenizer))
        
        trainer.train(train_loader, val_loader, EPOCH)
        print("Testing for Val dataset")
        trainer.eval(val_loader, save_file = "./output/Q3_"+c_name+"_val.txt")
        print("Testing for Test dataset")
        trainer.eval(test_loader, save_file = "./output/Q3_"+c_name+"_test.txt")
        trainer.save_model("./additional/models/")

    if infer:
        trainer.load_model("./additional/models/")
        trainer.test_sentence(crossing)
if __name__ == "__main__":
   main()