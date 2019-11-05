from mean_teacher.utils.utils_rao import generate_batches,initialize_double_optimizers,update_optimizer_state
from mean_teacher.modules.rao_datasets import RTEDataset
import time,os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm,tqdm_notebook
from torch.nn import functional as F
from mean_teacher.utils.logger import LOG
from mean_teacher.scorers.fnc_scorer import report_score


class Trainer():
    def __init__(self):
        self._current_time={time.strftime("%c")}

    def make_train_state(self,args):
        return {'stop_early': False,
                'early_stopping_step': 0,
                'early_stopping_best_val': 1e8,
                'learning_rate': args.learning_rate,
                'epoch_index': 0,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': [],
                'test_loss': -1,
                'test_acc': -1,
                'model_filename': args.model_state_file}

    def update_train_state(self,args, model, train_state):
        """Handle the training state updates.

        Components:
         - Early Stopping: Prevent overfitting.
         - Model Checkpoint: Model is saved if the model is better

        :param args: main arguments
        :param model: model to train
        :param train_state: a dictionary representing the training state values
        :returns:
            a new train_state
        """

        # Save one model at least
        if train_state['epoch_index'] == 0:
            torch.save(model.state_dict(), "model"+"_e"+str(train_state['epoch_index'])+".pth")
            train_state['stop_early'] = False
            assert type(train_state['val_acc']) is list
            all_val_acc_length=len(train_state['val_acc'])
            assert all_val_acc_length > 0
            acc_current_epoch = train_state['val_acc'][all_val_acc_length-1]
            train_state['early_stopping_best_val'] = acc_current_epoch

        # Save model if performance improved
        elif train_state['epoch_index'] >= 1:
            loss_tm1, acc_current_epoch = train_state['val_acc'][-2:]

            # If accuracy decreased
            if acc_current_epoch < train_state['early_stopping_best_val']:
                # increase patience counter
                train_state['early_stopping_step'] += 1
                LOG.info(f"found that acc_current_epoch  {acc_current_epoch} is less than or equal to the best dev "
                         f"accuracy value so far which is"
                         f" {train_state['early_stopping_best_val']}. "
                         f"Increasing patience total value. "
                         f"of patience now is {train_state['early_stopping_step']}")
            # accuracy increased
            else:
                # Save the best model
                torch.save(model.state_dict(), train_state['model_filename']+".pth")
                LOG.info(
                    f"found that acc_current_epoch loss {acc_current_epoch} is more than the best accuracy so far which is "
                    f"{train_state['early_stopping_best_val']}.resetting patience=0")
                # Reset early stopping step
                train_state['early_stopping_step'] = 0
                train_state['early_stopping_best_val']=acc_current_epoch

            # Stop early ?
            train_state['stop_early'] = \
                train_state['early_stopping_step'] >= args.early_stopping_criteria

        return train_state

    def get_argmax(self,predicted_labels):
        m = nn.Softmax()
        output_sftmax = m(predicted_labels)
        _, pred = output_sftmax.topk(1, 1, True, True)
        return pred.t()

    def accuracy_fever(self,predicted_labels, gold_labels):
        m = nn.Softmax()
        output_sftmax = m(predicted_labels)
        NO_LABEL = -1
        labeled_minibatch_size = max(gold_labels.ne(NO_LABEL).sum(), 1e-8)
        _, pred = output_sftmax.topk(1, 1, True, True)

        # gold labels and predictions are in transposes (eg:1x15 vs 15x1). so take a transpose to correct it.
        pred_t = pred.t()
        correct = pred_t.eq(gold_labels.view(1, -1).expand_as(pred_t))

        # take sum because in correct_k all the LABELS that match are now denoted by 1. So the sum means, total number of correct answers
        correct_k = correct.sum(1)
        correct_k_float = float(correct_k.data.item())
        labeled_minibatch_size_f = float(labeled_minibatch_size)
        result2 = (correct_k_float / labeled_minibatch_size_f) * 100


        return result2

    def compute_accuracy(self,y_pred, y_target):
        y_target = y_target.cpu()
        #y_pred_indices = (torch.sigmoid(y_pred) > 0.5).cpu().long()  # .max(dim=1)[1]
        n_correct = torch.eq(y_pred.long(), y_target).sum().item()
        return n_correct / len(y_pred) * 100

    def get_learning_rate(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def calculate_argmax_list(self, logit):
        list_labels_pred = []
        for tensor in logit:
            values, indices = torch.max(tensor, 0)
            list_labels_pred.append(indices.data.item())
        return list_labels_pred

    def train(self, args_in,classifier,dataset,comet_value_updater):
        classifier = classifier.to(args_in.device)

        if torch.cuda.is_available():
            class_loss_func = nn.CrossEntropyLoss(size_average=True).cuda()
        else:
            class_loss_func = nn.CrossEntropyLoss(size_average=True).cpu()

        input_optimizer, inter_atten_optimizer = initialize_double_optimizers(classifier, args_in)

        LOG.debug(f"going to get into ReduceLROnPlateau ")


        train_state_in = self.make_train_state(args_in)

        epoch_bar = tqdm_notebook(desc='training routine',
                                  total=args_in.num_epochs,
                                  position=0)

        dataset.set_split('train')
        train_bar = tqdm_notebook(desc='split=train',
                                  total=dataset.get_num_batches(args_in.batch_size),
                                  position=1,
                                  leave=True)
        dataset.set_split('val')
        val_bar = tqdm_notebook(desc='split=val',
                                total=dataset.get_num_batches(args_in.batch_size),
                                position=1,
                                leave=True)


        try:
            for epoch_index in range(args_in.num_epochs):
                train_state_in['epoch_index'] = epoch_index

                # Iterate over training dataset

                # setup: batch generator, set loss and acc to 0, set train mode on
                dataset.set_split('train')
                batch_generator1 = generate_batches(dataset,workers=args_in.workers,batch_size=args_in.batch_size,device=args_in.device)


                running_loss = 0.0
                running_acc = 0.0
                classifier.train()
                no_of_batches= int(len(dataset)/args_in.batch_size)




                for batch_index, batch_dict1 in enumerate(batch_generator1):

                    # the training routine is these 5 steps:

                    # --------------------------------------
                    # step 1. zero the gradients
                    input_optimizer.zero_grad()
                    inter_atten_optimizer.zero_grad()

                    #this code is from the libowen code base we are using for decomposable attention
                    if epoch_index == 0 and args_in.optimizer == 'adagrad':
                        update_optimizer_state(input_optimizer, inter_atten_optimizer, args_in)



                    # step 2. compute the output
                    y_pred_logit = classifier(batch_dict1['x_claim'], batch_dict1['x_evidence'])

                    # step 3. compute the loss
                    loss = class_loss_func(y_pred_logit, batch_dict1['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index + 1)
                    if (comet_value_updater is not None):
                        comet_value_updater.log_metric("running_loss_per_batch", running_loss,step=batch_index)

                    # step 4. use loss to produce gradients
                    loss.backward()




                    # step 5. use optimizer to take gradient step
                    #optimizer.step()
                    input_optimizer.step()
                    inter_atten_optimizer.step()

                    # -----------------------------------------
                    # compute the accuracy
                    

                    acc_t = self.accuracy_fever(y_pred_logit, batch_dict1['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index + 1)
                    if (comet_value_updater is not None):
                        comet_value_updater.log_metric("avg_accuracy_train_per_batch", running_acc, step=batch_index)

                    # update bar
                    train_bar.set_postfix(loss=running_loss,
                                          acc=running_acc,
                                          epoch=epoch_index)
                    train_bar.update()
                    LOG.info(f"epoch:{epoch_index} \t batch:{batch_index}/{no_of_batches} \t moving_avg_train_loss:{round(running_loss,2)} \t moving_avg_train_accuracy:{round(running_acc,2)} ")

                lr = self.get_learning_rate(input_optimizer)
                LOG.debug(f"value of learning rate now  for input_optimizer is:{lr}")
                lr = self.get_learning_rate(inter_atten_optimizer)
                LOG.debug(f"value of learning rate now  for inter_atten_optimizer is:{lr}")

                train_state_in['train_loss'].append(running_loss)
                train_state_in['train_acc'].append(running_acc)

                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("avg_accuracy_train_per_epoch", running_acc, step=epoch_index)

                # Iterate over val dataset

                # setup: batch generator, set loss and acc to 0; set eval mode on
                dataset.set_split('val')


                batch_generator1 = generate_batches(dataset,workers=args_in.workers,batch_size=args_in.batch_size,device=args_in.device,shuffle=False)

                running_loss = 0.
                running_acc = 0.
                classifier.eval()
                no_of_batches = int(len(dataset) / args_in.batch_size)

                for batch_index_dev, batch_dict1 in enumerate(batch_generator1):
                    # compute the output
                    y_pred_logit = classifier(batch_dict1['x_claim'], batch_dict1['x_evidence'])

                    # step 3. compute the loss
                    loss = class_loss_func(y_pred_logit, batch_dict1['y_target'])
                    loss_t = loss.item()
                    running_loss += (loss_t - running_loss) / (batch_index_dev + 1)



                    acc_t = self.accuracy_fever(y_pred_logit, batch_dict1['y_target'])
                    running_acc += (acc_t - running_acc) / (batch_index_dev + 1)

                    val_bar.set_postfix(loss=running_loss,
                                        acc=running_acc,
                                        epoch=epoch_index)
                    val_bar.update()
                    LOG.info(
                        f"epoch:{epoch_index} \t batch:{batch_index_dev}/{no_of_batches} \t moving_avg_val_loss:{round(running_loss,2)} \t moving_avg_val_accuracy:{round(running_acc,2)} ")
                    if(comet_value_updater is not None):
                        comet_value_updater.log_metric("avg_accuracy_dev_per_batch", running_acc, step=batch_index_dev)

                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("running_dev_loss_per_epoch", running_loss, step=epoch_index)

                if (comet_value_updater is not None):
                    comet_value_updater.log_metric("avg_accuracy_dev_per_epoch", running_acc, step=epoch_index)

                train_state_in['val_loss'].append(running_loss)
                train_state_in['val_acc'].append(running_acc)

                train_state_in = self.update_train_state( args=args_in, model=classifier,
                                                      train_state=train_state_in)

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                if train_state_in['stop_early']:
                     break

                train_bar.n = 0
                val_bar.n = 0
                epoch_bar.update()

                LOG.info(f"epoch:{epoch_index}\tval_loss_end_of_epoch:{round(running_loss,4)}\tval_accuracy_end_of_epoch:{round(running_acc,4)} ")


            LOG.info(f"{self._current_time:}Val loss at end of all epochs: {(train_state_in['val_loss'])}")
            LOG.info(f"{self._current_time:}Val accuracy at end of all epochs: {(train_state_in['val_acc'])}")

        except KeyboardInterrupt:
            print("Exiting loop")

    def get_label_strings_given_vectorizer(self, vectorizer, predictions_index_labels):
        labels_str=[]
        for e in predictions_index_labels[0]:
            labels_str.append(vectorizer.label_vocab.lookup_index(e.item()).lower())
        return labels_str


    def get_label_strings_given_list(self, labels_tensor):
        LABELS = ['agree', 'disagree', 'discuss', 'unrelated']
        labels_str=[]
        for e in labels_tensor:
            labels_str.append(LABELS[e.item()].lower())
        return labels_str

    def test(self, args_in,classifier, dataset,split_to_test,vectorizer):
        if(args_in.load_model_from_disk):
            assert os.path.exists(args_in.trained_model_path) is True
            assert os.path.isfile(args_in.trained_model_path) is True
            if os.path.getsize(args_in.trained_model_path) > 0:
                classifier.load_state_dict(torch.load(args_in.trained_model_path,map_location=torch.device(args_in.device)))
        classifier.eval()
        dataset.set_split(split_to_test)
        batch_generator1 = generate_batches(dataset, workers=args_in.workers, batch_size=args_in.batch_size,
                                            device=args_in.device, shuffle=False)

        running_loss = 0.
        running_acc = 0.

        no_of_batches = int(len(dataset) / args_in.batch_size)
        total_predictions=[]
        total_gold = []

        for batch_index_dev, batch_dict in enumerate(batch_generator1):
            # compute the output
            y_pred_logit = classifier(batch_dict['x_claim'], batch_dict['x_evidence'])
            if torch.cuda.is_available():
                class_loss_func = nn.CrossEntropyLoss(size_average=True).cuda()
            else:
                class_loss_func = nn.CrossEntropyLoss(size_average=True).cpu()

            # compute the loss
            loss = class_loss_func(y_pred_logit, batch_dict['y_target'])
            loss_t = loss.item()
            running_loss += (loss_t - running_loss) / (batch_index_dev + 1)

            acc_t=0

            if(args_in.database_to_test_with=="fnc"):
                predictions_index_labels=self.get_argmax(y_pred_logit.float())
                predictions_str_labels=self.get_label_strings_given_vectorizer(vectorizer, predictions_index_labels)
                gold_str=self.get_label_strings_given_list(batch_dict['y_target'])
                for e in gold_str:
                    total_gold.append(e)
                for e in predictions_str_labels:
                    total_predictions.append(e)
            else:
                acc_t = self.accuracy_fever(y_pred_logit, batch_dict['y_target'])
            running_acc += (acc_t - running_acc) / (batch_index_dev + 1)
            LOG.info(
                f" \t batch:{batch_index_dev}/{no_of_batches} \t moving_avg_val_loss:{round(running_loss,2)} \t moving_avg_val_accuracy:{round(running_acc,2)} ")


        if (args_in.database_to_test_with == "fnc"):
            running_acc = report_score(total_gold, total_predictions)

        train_state_in = self.make_train_state(args_in)
        train_state_in['test_loss'] = running_loss
        train_state_in['test_acc'] = running_acc

        LOG.info(f" test_accuracy : {(train_state_in['test_acc'])}")
        print(f" test_accuracy : {(train_state_in['test_acc'])}")


