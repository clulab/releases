import sys
from pathlib import Path
import argparse

parent_folder_path = str(Path('.').absolute().parent)
models_folder_path = parent_folder_path+"/models"
data_folder_path = parent_folder_path+"/data"
sys.path+=[parent_folder_path, data_folder_path, models_folder_path]

import random
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch import optim
import torch
import os

import csv

class Seq2SeqT5Experiment():
    def __init__(self, learning_rate, device):
        self.tokenizer = T5Tokenizer.from_pretrained('t5-small')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-small')
        self.t5_model.to(device)

        # This learning rate 0.0001 is used in one of the tutorial, but might not be the best choice.
        self.adamOpt = optim.Adam(self.t5_model.parameters(), lr=learning_rate)
        self.device = device

    def train_iters(self, train_pairs, n_iters, print_every=10):
        self.t5_model.train()

        print_loss_total = 0  # Reset every print_every

        # Training data is checked to be correct.
        training_pair_indices = [random.choice(range(len(train_pairs))) for i in range(n_iters)]

        for iter, idx in enumerate(training_pair_indices):
            self.adamOpt.zero_grad()

            training_pair = train_pairs[idx]
            input_tensor = self.tokenizer.encode(training_pair["input"], return_tensors="pt").to(self.device)
            target_tensor = self.tokenizer.encode(training_pair["output"], return_tensors="pt").to(self.device)

            outputs = self.t5_model(input_ids=input_tensor, labels=target_tensor)
            loss, prediction_scores = outputs[:2]

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("iter ", iter, " average loss:",print_loss_avg)
                print_loss_total = 0

    def evaluate_iters(self, test_pairs):
        self.t5_model.eval()
        with torch.no_grad():
            for i in range(len(test_pairs)):
                input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)

                predicted_tensor = self.t5_model.generate(input_tensor)

                print("-"*30)
                print("input sent:"+test_pairs[i]["input"])
                print("\ttarget sent:"+test_pairs[i]["output"])
                print("\tpredict sent:"+self.tokenizer.decode(predicted_tensor[0]))


    def evaluate_iters_and_get_loss(self, test_pairs, print_every = 200):
        self.t5_model.eval()

        total_loss = 0
        with torch.no_grad():
            for i in range(len(test_pairs)):
                input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)
                target_tensor = self.tokenizer.encode(test_pairs[i]["output"], return_tensors="pt").to(self.device)

                loss = self.t5_model(input_ids=input_tensor, labels=target_tensor)[0]
                total_loss+=loss.detach().cpu().numpy()

                if i%print_every==0:
                    print("evaluating "+ str(i)+  " out of " +str(len(test_pairs)) )

            print("average loss:", total_loss/len(test_pairs))

        return total_loss/len(test_pairs)

    def evaluate_iters_and_save_output(self, test_pairs, avg_loss, output_tsv_file):
        print("="*30)
        print("saving output ...")

        with open(output_tsv_file, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(['input', 'target', 'output', 'avg loss:'+str(avg_loss)])

            self.t5_model.eval()
            with torch.no_grad():
                for i in range(len(test_pairs)):
                    input_tensor = self.tokenizer.encode(test_pairs[i]["input"], return_tensors="pt").to(self.device)
                    predicted_tensor = self.t5_model.generate(input_tensor)
                    tsv_writer.writerow([test_pairs[i]["input"], test_pairs[i]["output"], self.tokenizer.decode(predicted_tensor[0])])


    def save_tuned_model(self, save_model_name):

        torch.save(self.t5_model, save_model_name)

