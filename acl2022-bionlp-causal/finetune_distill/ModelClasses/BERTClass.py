from transformers import BertForSequenceClassification, BertTokenizer
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, recall_score
import time

class BERTExperiment:
    def __init__(self, device, lr=0.00002, n_epoch=20, model_opt="BioBERT"):
        assert(lr == 0.00002)

        # Official doc: https://huggingface.co/transformers/model_doc/bert.html#bertforsequenceclassification

        # according to this: https://stackoverflow.com/questions/60876394/does-bertforsequenceclassification-classify-on-the-cls-vector
        # BertForSequenceClassification is what we need. It uses the CLS embedding for sequence classification (not average pooling).
        if model_opt == "BERT":
            print("loading bert base uncased model ...")
            self.bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        if model_opt == "BioBERT":
            print("loading bio bert base cased model ...")
            self.bert_classifier = BertForSequenceClassification.from_pretrained('dmis-lab/biobert-base-cased-v1.1', num_labels=3)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        # self.bert_classifier = BertForSequenceClassification.from_pretrained('bert-base-cased',
        #                                                                      num_labels=3)
        if model_opt == "TinyBERT":
            print("loading mini bert base uncased model ...")
            self.bert_classifier = BertForSequenceClassification.from_pretrained("google/bert_uncased_L-8_H-128_A-2", num_labels=3)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if model_opt == "MiniBERT":
            self.bert_classifier = BertForSequenceClassification.from_pretrained("google/bert_uncased_L-4_H-256_A-4",
                                                                                 num_labels=3)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if model_opt == "BioTinyBERT":
            self.bert_classifier = BertForSequenceClassification.from_pretrained("/work/zhengzhongliang/2021_MiniBERT_Pretrain/2021_MiniBERT_Pubmed/saved_models/model_google/bert_uncased_L-8_H-128_A-2_tokenizer_bert-base-uncased_12_epochs_split_by_sent",
                                                                                 num_labels=3)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Bio MiniBERT: mini bert trained with MLM on some pubmed publications
        if model_opt == "BioMiniBERT":
            print("loading bio mini bert base uncased model ...")
            self.bert_classifier = BertForSequenceClassification.from_pretrained("/work/zhengzhongliang/2021_MiniBERT_Pretrain/2021_MiniBERT_Pubmed/saved_models/model_google/bert_uncased_L-4_H-256_A-4_tokenizer_bert-base-uncased_12_epochs_split_by_sent",
                                                                                 num_labels=3)
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        if model_opt == "TinyBERTPubmed":  # Bert L8 H128 with reduced vocab, pretrained on 10000 pubmed papers
            self.bert_classifier = BertForSequenceClassification.from_pretrained(
                "/work/zhengzhongliang/2021_MiniBERT_Pretrain/2021_MiniBERT_Pubmed/saved_models/model_google/bert_uncased_L-8_H-128_A-2_tokenizer_vocab_pubmed_10000_12_epochs_split_by_sent_better_resize",
                num_labels=3
            )
            self.tokenizer = BertTokenizer.from_pretrained("/work/zhengzhongliang/2021_MiniBERT_Pretrain/2021_MiniBERT_Pubmed/saved_models/model_google/bert_uncased_L-8_H-128_A-2_tokenizer_vocab_pubmed_10000_12_epochs_split_by_sent_better_resize")

        if model_opt == "MiniBERTPubmed":  # Bert L4 H256 with reduced vocab, pretrained on 10000 pubmed papers
            self.bert_classifier = BertForSequenceClassification.from_pretrained(
                "/work/zhengzhongliang/2021_MiniBERT_Pretrain/2021_MiniBERT_Pubmed/saved_models/model_google/bert_uncased_L-4_H-256_A-4_tokenizer_vocab_pubmed_10000_12_epochs_split_by_sent_better_resize",
                num_labels=3
            )
            self.tokenizer = BertTokenizer.from_pretrained(
                "/work/zhengzhongliang/2021_MiniBERT_Pretrain/2021_MiniBERT_Pubmed/saved_models/model_google/bert_uncased_L-4_H-256_A-4_tokenizer_vocab_pubmed_10000_12_epochs_split_by_sent_better_resize")

        self.bert_classifier.to(device)

        self.n_epoch = n_epoch
        self.adamOpt = torch.optim.Adam(params=self.bert_classifier.parameters(), lr=lr)

        self.criterion = nn.CrossEntropyLoss()
        self.distill_criterion = torch.nn.MSELoss()

        self.device = device

    def save_classifier(self):

        return 0

    def train_epoch(self, batches, tokenizer, debug_flag=False, print_every=20):
        self.bert_classifier.train()

        if debug_flag:
            print("="*20)

        print_loss_total = 0
        for batch_idx, batch in enumerate(batches):
            self.adamOpt.zero_grad()

            loss, logits = self.bert_classifier(input_ids=torch.tensor(batch["input_ids"], dtype = torch.int64).to(self.device),
                                                attention_mask=torch.tensor(batch["attention_mask"], dtype = torch.int64).to(self.device),
                                                token_type_ids=torch.tensor(batch["token_type_ids"], dtype = torch.int64).to(self.device),
                                                labels=torch.tensor(batch["target"], dtype = torch.int64).to(self.device))

            if debug_flag:
                print("\tlogits:", logits, " target:", batch["target"], " loss:", loss)

                if batch_idx < 3:
                    print("\ttokens:", batch["tokens"])
                    print("\tindices:", batch["input_ids"])
                    print("\tatt masks:", batch["attention_mask"])
                    print("\ttoken type ids:", batch["token_type_ids"])
                    print("\treconstructed token:", [tokenizer.convert_ids_to_tokens(indices) for indices in batch["input_ids"]])
                    print("\n wait for next example")

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if batch_idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("\tbatch num:", batch_idx, " average loss:", print_loss_avg)
                print_loss_total = 0

        if debug_flag:
            input("-"*20)

        return 0

    def eval_epoch(self, batches, tokenizer, debug_flag=False):
        self.bert_classifier.eval()

        if debug_flag:
            print("-"*20)

        label_list = []
        pred_list = []
        time_record = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(batches):
                self.adamOpt.zero_grad()

                if self.device != torch.device("cpu"):  # This is important!
                    torch.cuda.synchronize()
                start_time = time.time()
                bert_outputs = self.bert_classifier(input_ids=torch.tensor(batch["input_ids"], dtype=torch.int64).to(self.device),
                                                attention_mask=torch.tensor(batch["attention_mask"], dtype=torch.int64).to(self.device),
                                                token_type_ids=torch.tensor(batch["token_type_ids"], dtype=torch.int64).to(self.device))

                if self.device != torch.device("cpu"):  # This is important!
                    torch.cuda.synchronize()
                end_time = time.time()
                time_record.append(end_time - start_time)

                pred_labels = torch.max(bert_outputs[0], dim=1)[1]

                if debug_flag:
                    print("\tlogits:", bert_outputs[0], " pred:", pred_labels, " target:", batch["target"])

                    if batch_idx < 3:
                        print("\ttokens:", batch["tokens"])
                        print("\tindices:", batch["input_ids"])
                        print("\tatt masks:", batch["attention_mask"])
                        print("\ttoken type ids:", batch["token_type_ids"])
                        print("\treconstructed token:", [tokenizer.convert_ids_to_tokens(indices) for indices in batch["input_ids"]])
                        print("\n wait for next example")

                label_list.extend(batch["target"].cpu().tolist())
                pred_list.extend(pred_labels.detach().cpu().tolist())

            precision, recall, f1 = self.calculate_p_r_f1(label_list, pred_list)
            print("\tp:", precision, "r:", recall, "f1:", f1)

        if debug_flag:
            input("-"*20)

        return f1, label_list, pred_list, time_record

    def train_distill_epoch(self, dataset_train, tokenizer, print_every=20,
                      debug_flag=False):
        self.bert_classifier.train()

        print_loss_total = 0
        for batch_idx, batch in enumerate(dataset_train):
            self.adamOpt.zero_grad()

            _, logits = self.bert_classifier(
                input_ids=torch.tensor(batch["input_ids"], dtype=torch.int64).to(self.device),
                attention_mask=torch.tensor(batch["attention_mask"], dtype=torch.int64).to(self.device),
                token_type_ids=torch.tensor(batch["token_type_ids"], dtype=torch.int64).to(self.device),
                labels=torch.tensor(batch["target"], dtype=torch.int64).to(self.device))

            loss = self.distill_criterion(logits, batch["teacher_pred_scores"].to(self.device))

            if debug_flag:
                print("=" * 20)
                print("\ttokens:", batch["tokens"])
                print("\tindices:", batch["input_ids"])
                print("\tatt masks:", batch["attention_mask"])
                print("\ttoken type ids:", batch["token_type_ids"])
                print("\tlogits:", logits)
                print("\tteacher pred scores:", batch["teacher_pred_scores"])
                print("\tloss:", loss)
                input("-" * 40)

            loss.backward()
            self.adamOpt.step()

            print_loss_total += loss.detach().cpu().numpy()

            if batch_idx % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print("\tbatch num:", batch_idx, " average loss:", print_loss_avg)
                print_loss_total = 0

        if debug_flag:
            input("-" * 20)

        return 0


    def eval_manual(self):


        return 0

    @classmethod
    def calculate_p_r_f1(cls, label_list, pred_list):
        tp = 0
        fp = 0
        fn = 0

        smooth = 1e-7

        for i in range(len(label_list)):
            if pred_list[i] != 0 and label_list[i] == pred_list[i]:
                tp += 1

            if pred_list[i] != 0 and label_list[i] != pred_list[i]:
                fp += 1

            if pred_list[i] == 0 and label_list[i] != pred_list[i]:
                fn += 1

        precision = tp / (tp + fp + smooth)
        recall = tp / (tp + fn + smooth)
        f1 = 2 * precision * recall / (precision + recall + smooth)

        return precision, recall, f1