import torch
from transformers import BertModel, AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
from dataset import *
from datasets import load_dataset, load_metric
from functools import partial
import os
import argparse

class STSBERTModel(torch.nn.Module):
  """
  Bert-tiny Model with a linear layer at the end to predict the value of similarity score
  """
  def __init__(self, pre_trained_model_name):
    """
    Init a bert model and a linear layer with a dropout layer in between
    """
    super(STSBERTModel, self).__init__()
    self.bert_model = BertModel.from_pretrained(pre_trained_model_name, return_dict=False)
    self.bert_drop = torch.nn.Dropout(0.3)
    self.fnn = torch.nn.Linear(128, 1)

  def forward(self, input_ids, attention_mask, token_type_ids):
    """
    Pass the input sent1 and sent2 together to bert model and use the pooled_output from it to predict similarity score
    """
    _, pooled_output = self.bert_model(input_ids = input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)
    output = self.fnn(self.bert_drop(pooled_output))
    return output

def all_to_all_collate_fn(data, tokenizer):
    """
    A helper function to create a batch of encoded sentence so that it can be passed to the bert model
    """
    input_sents = [i[0] for i in data]
    labels = [i[1] for i in data]
    input_encoded = tokenizer.batch_encode_plus(input_sents, padding="max_length", max_length = 32, truncation=True, return_tensors='pt')
    return input_encoded, labels

class STSBERTModelTrainer:
    """
    A class that can help in finetuning the bert-tiny for the STS task.
    """
    def __init__(self,pre_trained_model_name, device):
        """
        Initialise the STSBERTModel along with tokenizer and optimiser
        """
        self.task = "stsb"
        self.device = device
        self.model = STSBERTModel(pre_trained_model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(pre_trained_model_name)
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)
        self.metric = load_metric('glue', self.task)
        return
    def freeze_bert(self,_freeze=True):
        """
        Method to freeze the bert params for some of the epochs.
        """
        for param in self.model.bert_model.parameters():
            param.requires_grad = not _freeze

    def train(self, train_loader, val_loader, EPOCH, fz_bert_epoch):
        """
        Method to train the model.
        For first fx_ber_epoch freeze the bert params and only train the params in linear layer.
        """
        self.freeze_bert()
        for epoch in range(EPOCH):
            print("Epoch ", epoch, ":")
            if epoch == fz_bert_epoch:
                self.freeze_bert(_freeze = False)
                for g in self.optimizer.param_groups:
                    g['lr'] = 1e-5
            self.model.train()
            train_loss = 0
            for inputs, labels in tqdm(train_loader):
                input_ids = torch.tensor(inputs['input_ids']).to(self.device)
                input_attention_mask = torch.tensor(inputs['attention_mask']).to(self.device)
                input_token_type_ids = torch.tensor(inputs['token_type_ids']).to(self.device)
                labels = torch.tensor(labels).to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(input_ids = input_ids, attention_mask = input_attention_mask, token_type_ids = input_token_type_ids)
                loss = torch.nn.functional.mse_loss(outputs, labels.view(-1, 1))
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)
            print("Train Loss:", train_loss)
            val_loss, (targets, preds) = self.eval(val_loader)
        return
    
    def eval(self, val_loader, save_file = None):
        """
        Method to perform evaluation on the model.
        """
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            predictions = []
            targets = []
            for inputs, labels in val_loader:
                input_ids = torch.tensor(inputs['input_ids']).to(self.device)
                input_attention_mask = torch.tensor(inputs['attention_mask']).to(self.device)
                input_token_type_ids = torch.tensor(inputs['token_type_ids']).to(self.device)
                labels = torch.tensor(labels).to(self.device)
                outputs = self.model(input_ids = input_ids, attention_mask = input_attention_mask, token_type_ids = input_token_type_ids)
                loss = torch.nn.functional.mse_loss(outputs, labels.view(-1, 1))
                val_loss += loss.item()
                outputs = outputs.squeeze()
                predictions.extend(outputs.tolist())
                targets.extend(labels)
            val_loss /= len(val_loader)
            print("Val Loss:", val_loss)
            print(predictions)
            print(targets)
            print("Sim Metric:", self.metric.compute(predictions=predictions, references=targets))
            if save_file:
                print("Saving outputs in ", save_file)
                with open(save_file,'w') as ofile:
                    for item, _ in enumerate(predictions):
                        ofile.write(str(predictions[item])+"\n")
        return val_loss, (targets, predictions)

    def save_model(self, model_dir = None):
        """
        Method to save the model
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        file_name = "all_to_all_bert"
        file_name = model_dir + file_name + ".pt"

        print("saving model into ", file_name)
        torch.save(self.model.state_dict(),
                file_name)
        return
    
    def load_model(self, model_dir = None):
        """
        method to load the model.
        """
        file_name = "all_to_all_bert"
        file_name = model_dir + file_name + ".pt"

        print("Loading from ", file_name)
        self.model.load_state_dict(torch.load(file_name))
        return 

    def test_sentence(self):
        """
        Method to perform inference on the sentences.
        """
        print("Press Ctrl+C to end this prompt:")
        while True:
            sent1 = str(input("Enter Sentence 1: "))
            sent2 = str(input("Enter Sentence 2: "))
            print("Sent1: ",sent1)
            print("Sent2: ",sent2)
            encodded_sent = self.tokenizer.encode_plus((sent1,sent2), return_tensors='pt')
            print(encodded_sent)
            self.model.eval()
            with torch.no_grad():
                sim_score = self.model(input_ids = encodded_sent['input_ids'], attention_mask = encodded_sent['attention_mask'], token_type_ids = encodded_sent['token_type_ids'])
                print("Sim Score:", sim_score)
                
def main():
    parser = argparse.ArgumentParser(description="STS using BERT-Tiny",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--train", action='store_true', help="Whether to train the model")
    parser.add_argument("-i", "--infer", action='store_true', help="Whether to infer using command line")
    parser.add_argument("-e", "--epoch", default=30, help="number of epoch to train")
    parser.add_argument("-b", "--batchsz", default=16, help="Size of the batch to use while training")
    args = vars(parser.parse_args())
    
    BATCH_SIZE = int(args["batchsz"])
    EPOCH = int(args["epoch"])
    train = args["train"]
    infer = args["infer"]

    task = "stsb"
    fz_bert_epoch = 10
    seed = 42
    pre_trained_model_name = "prajjwal1/bert-tiny"

    torch.manual_seed(seed)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print("Using device: ",device)


    trainer = STSBERTModelTrainer(pre_trained_model_name, device)
    
    if train:
        print("Finetuning the Model")
        dataset = load_dataset("glue", task)
        train_dataset = STSDataset(dataset["train"])
        val_dataset = STSDataset(dataset["validation"])
        test_dataset = STSDataset(dataset["test"])
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn =partial(all_to_all_collate_fn, tokenizer = trainer.tokenizer))
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,collate_fn = partial(all_to_all_collate_fn, tokenizer = trainer.tokenizer))
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,collate_fn = partial(all_to_all_collate_fn, tokenizer = trainer.tokenizer))
        
        trainer.load_model(model_dir="./additional/models/")
        #trainer.train(train_loader, val_loader, EPOCH, fz_bert_epoch)
        #print("Testing for Validation dataset")
        #trainer.eval(val_loader, "./output/Q1_val.txt")
        print("Testing for Test dataset")
        trainer.eval(test_loader, "./output/Q1_test.txt")
        trainer.save_model("./additional/models/")

    if infer:
        print("Infering from the model")
        trainer.load_model(model_dir="./additional/models/")
        trainer.test_sentence()

if __name__ == "__main__":
   main()