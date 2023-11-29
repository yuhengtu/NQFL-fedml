import copy
import logging
import random

import numpy as np
import torch
import wandb
import pandas as pd

from fedml_api.standalone.fedavg_our.client import Client

from fedml_api.standalone.fedavg_our_v2.communication_rate_function_multi_variables_v2 import RDfucntion


class FedAvgAPI(object):
    def __init__(self, dataset, device, args, model_trainer):
        self.device = device
        self.args = args
        [train_data_num, test_data_num, train_data_global, test_data_global,
         train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num

        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict

        self.model_trainer = model_trainer
        self._setup_clients(train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer)

    def _setup_clients(self, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, model_trainer):
        logging.info("############setup_clients (START)#############")
        for client_idx in range(self.args.client_num_per_round):
            c = Client(client_idx, train_data_local_dict[client_idx], test_data_local_dict[client_idx],
                       train_data_local_num_dict[client_idx], self.args, self.device, model_trainer)
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")

    def train(self):
        # self.args.frequency_of_the_test = 1
        # w_global = self.model_trainer.get_model_params()
        w_global = self.model_trainer.get_model_params_cuda()

        # preparation
        sz_t = self.args.comm_round
        sz_k = self.args.client_num_per_round
        n = self._get_param_size(w_global)

        G_global  = np.zeros((sz_t + 1, n))    # 全局的梯度值
        G_client  = np.zeros((sz_t,sz_k, n))   # 本地的梯度值
        Var_S     = np.zeros((sz_t + 1, n))              # 全局的梯度方差    ###
        Var_G     = np.zeros((sz_t + 1,sz_k, n))       # 本地的梯度方差
        Mean_S    = np.zeros(n)
        D         = np.full(sz_t + 1, 0.00005)   # 每次迭代允许的失真约束
        S_X       = np.zeros((sz_t + 1, sz_k, n))    # 本地对全局的估计
        M         = np.zeros((sz_t + 1, sz_k, n))    # 本地对全局的估计的误差
        nu_real   = np.zeros((sz_t + 1, n))           # 注水线
        P         = np.zeros((sz_t + 1, sz_k, n))   # 每个client的压缩误差
        self.R         = np.zeros((sz_t + 1, n))           # 和速率

        G_value = np.zeros((sz_t + 1, sz_k, n))
        R_client = np.zeros((sz_t + 1, sz_k, n))
        # initialization
        G_global[0] = np.abs(np.random.normal(0, Var_S[0]))

        S_X[0] = 0.4
        Var_S[0] = 0.3

        

        Var_W    = 0.02

        # print('*'*20, w_global[w_global.keys()[0]].device)
        for round_idx in range(1, self.args.comm_round + 1):

            logging.info("################Communication round : {}".format(round_idx))

            w_locals = []
            g_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(round_idx, self.args.client_num_in_total,
                                                   self.args.client_num_per_round)
            logging.info("client_indexes = " + str(client_indexes))

            for idx, client in enumerate(self.client_list):
                # update dataset
                client_idx = client_indexes[idx]
                client.update_local_dataset(client_idx, self.train_data_local_dict[client_idx],
                                            self.test_data_local_dict[client_idx],
                                            self.train_data_local_num_dict[client_idx])

                # train on new dataset
                # w = client.train(copy.deepcopy(w_global))
                g, var_g = client.train(copy.deepcopy(w_global))
                # self.logger.info("local weights = " + str(w))
                # w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                g_locals.append((client.get_sample_number(), copy.deepcopy(g)))
                
                # 每轮训练的本地梯度方差
                Var_G[round_idx - 1, idx] = var_g.cpu().numpy()
            

            # update global weights
            # w_global = self._aggregate(w_locals)
            # g_global = self._aggregate_g(g_locals)

            # 将Var_S初始化为第一轮训练得到的全局梯度方差
            if round_idx == 1:
                Var_S[round_idx - 1] = self._get_grad_global_vars(g_locals)
                for k in range(0, sz_k):
                    M[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.1))
                    P[0, k] = M[0, k] + np.abs(np.random.normal(0, 0.2))
                for k in range(0, sz_k):
                    Var_G[0, k] = Var_S[0] + np.abs(np.random.normal(0, 0.5))
                    G_value[0, k] = np.random.normal(0, Var_G[0, k])
            else:
                Var_S[round_idx - 1] = Var_S[round_idx - 2] + Var_W
            # print('*' * 50, '全局梯度方差')
            # print(Var_S[round_idx - 1])
            # print(pd.DataFrame(Var_S[round_idx - 1]).describe())
            
            max_acc = 0
            best_model = None
            for i in range(100):
                G_global_tmp = np.random.normal(0, np.maximum(Var_S[round_idx - 1] - D[round_idx], 0))
                g_global = self._grad_shape_trans(G_global_tmp, w_global, self.device)
                cur_model = self._update_global_model(copy.deepcopy(w_global), g_global, self.args.lr)
                self.model_trainer.set_model_params(cur_model)
                cur_acc, _ = self._get_test_accuracy()
                if cur_acc > max_acc:
                    max_acc = cur_acc
                    best_model = cur_model
                    G_global[round_idx - 1] = G_global_tmp
            
            
            self.R[round_idx], P[round_idx, :], nu_real[round_idx],  S_X[round_idx, :], M[round_idx, :], R_client[round_idx, :] = RDfucntion(round_idx, sz_k, n, Var_S[round_idx - 1], Var_G[round_idx - 1, :], D[round_idx - 1], S_X[round_idx - 1, :], M[round_idx - 1, :], G_global[round_idx - 1], P[round_idx - 1, :], Var_W,  G_value[round_idx, :])

            w_global = best_model

            # g_global = self._grad_shape_trans(G_global[round_idx - 1], w_global, self.device)
            # 更新全局模型
            # self._update_global_model(w_global, g_global, self.args.lr)
            # 将server模型更新给client模型
            self.model_trainer.set_model_params(w_global)

            # test results
            # at last round
            if round_idx == self.args.comm_round:
                self._local_test_on_all_clients(round_idx)
            # per {frequency_of_the_test} round
            elif round_idx % self.args.frequency_of_the_test == 0:
                if self.args.dataset.startswith("stackoverflow"):
                    self._local_test_on_validation_set(round_idx)
                else:
                    self._local_test_on_all_clients(round_idx)

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        test_data_num = len(self.test_global.dataset)
        sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
        subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
        sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
        self.val_global = sample_testset

    def _aggregate(self, w_locals):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w
        return averaged_params
    
    def _aggregate_g(self, g_locals):
        training_num = 0
        for idx in range(len(g_locals)):
            (sample_num, averaged_gradients) = g_locals[idx]
            training_num += sample_num

        (sample_num, averaged_gradients) = g_locals[0]
        for k in range(len(averaged_gradients)):
            for i in range(0, len(g_locals)):
                local_sample_number, local_model_gradients = g_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_gradients[k] = local_model_gradients[k] * w
                else:
                    averaged_gradients[k] += local_model_gradients[k] * w
        return averaged_gradients

    def _update_global_model(self, model, gradients, lr):
        keys = list(model.keys())
        for i in range(len(keys)):
            model[keys[i]].data.add_(-lr, gradients[i].data)
        return model

    def _get_param_size(self, params):
        size = 0
        keys = params.keys()
        for k in keys:
            size += params[k].reshape(-1).shape[0]
        return size

    def _get_grad_global_vars(self, g_locals):
        g_total = None
        for g_local in g_locals:
            g_cur = g_local[1][0].reshape(-1)
            for i in range(1, len(g_local[1])):
                g_cur = torch.cat((g_cur, g_local[1][i].reshape(-1)))
            if g_total is None:
                g_total = g_cur.unsqueeze(0)
            else:
                g_total = torch.cat((g_total, g_cur.unsqueeze(0)))
        return g_total.var(0).cpu().numpy()

    def _grad_shape_trans(self, G_global_one, w_global, device):
        g_global = []
        keys = list(w_global.keys())
        cur_i = 0
        for k in keys:
            g_cur = G_global_one[cur_i:cur_i + w_global[k].reshape(-1).shape[0]].reshape(w_global[k].shape)
            cur_i += w_global[k].reshape(-1).shape[0]
            g_global.append(torch.tensor(g_cur).to(device))
        return g_global

    def _get_test_accuracy(self):
        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }
        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            if self.args.ci == 1:
                break
        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])
        return test_acc, test_loss

    def _local_test_on_all_clients(self, round_idx):

        logging.info("################local_test_on_all_clients : {}".format(round_idx))

        train_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        test_metrics = {
            'num_samples': [],
            'num_correct': [],
            'losses': []
        }

        client = self.client_list[0]

        for client_idx in range(self.args.client_num_in_total):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client.update_local_dataset(0, self.train_data_local_dict[client_idx],
                                        self.test_data_local_dict[client_idx],
                                        self.train_data_local_num_dict[client_idx])
            # train data
            train_local_metrics = client.local_test(False)
            train_metrics['num_samples'].append(copy.deepcopy(train_local_metrics['test_total']))
            train_metrics['num_correct'].append(copy.deepcopy(train_local_metrics['test_correct']))
            train_metrics['losses'].append(copy.deepcopy(train_local_metrics['test_loss']))

            # test data
            test_local_metrics = client.local_test(True)
            test_metrics['num_samples'].append(copy.deepcopy(test_local_metrics['test_total']))
            test_metrics['num_correct'].append(copy.deepcopy(test_local_metrics['test_correct']))
            test_metrics['losses'].append(copy.deepcopy(test_local_metrics['test_loss']))

            """
            Note: CI environment is CPU-based computing. 
            The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
            """
            if self.args.ci == 1:
                break

        # test on training dataset
        train_acc = sum(train_metrics['num_correct']) / sum(train_metrics['num_samples'])
        train_loss = sum(train_metrics['losses']) / sum(train_metrics['num_samples'])

        # test on test dataset
        test_acc = sum(test_metrics['num_correct']) / sum(test_metrics['num_samples'])
        test_loss = sum(test_metrics['losses']) / sum(test_metrics['num_samples'])

        # communication bits


        stats = {'training_acc': train_acc, 'training_loss': train_loss}
        wandb.log({"Train/Acc": train_acc, "round": round_idx})
        wandb.log({"Train/Loss": train_loss, "round": round_idx})
        logging.info(stats)

        stats = {'test_acc': test_acc, 'test_loss': test_loss}
        wandb.log({"Test/Acc": test_acc, "round": round_idx})
        wandb.log({"Test/Loss": test_loss, "round": round_idx})
        logging.info(stats)

        # print(self.R[:round_idx + 1])
        # print(np.sum(self.R[0]))
        # print(np.sum(self.R[round_idx]))
        
        # communication bits - accuracy
        tmp_r = self.R[:round_idx + 1]
        cb = np.sum(tmp_r[~np.isnan(tmp_r)])
        stats = {'communication bits': cb}
        wandb.log({"Train/Acc": train_acc, "communication bits": cb})
        wandb.log({"Test/Acc": test_acc, "communication bits": cb})
        logging.info(stats)

    def _local_test_on_validation_set(self, round_idx):

        logging.info("################local_test_on_validation_set : {}".format(round_idx))

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics['test_correct'] / test_metrics['test_total']
            test_pre = test_metrics['test_precision'] / test_metrics['test_total']
            test_rec = test_metrics['test_recall'] / test_metrics['test_total']
            test_loss = test_metrics['test_loss'] / test_metrics['test_total']
            stats = {'test_acc': test_acc, 'test_pre': test_pre, 'test_rec': test_rec, 'test_loss': test_loss}
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Pre": test_pre, "round": round_idx})
            wandb.log({"Test/Rec": test_rec, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception("Unknown format to log metrics for dataset {}!" % self.args.dataset)

        logging.info(stats)
