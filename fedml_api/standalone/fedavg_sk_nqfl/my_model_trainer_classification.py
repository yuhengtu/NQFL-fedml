import logging

import torch
from torch import nn
from scipy.stats import norm
import numpy as np

try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()
    
    def get_model_params_cuda(self, device):
        return self.model.to(device).state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    def get_model_gradients(self):
        return self.grad_accum
    
    #变
    def get_comm_bits(self, q):
        cb = 0
        for i in range(len(self.grad_accum)):
            d = self.grad_accum[i].reshape(-1).shape[0]
            cur_cb = d * np.ceil(np.log2(q)) + 32 + 32
            cb += cur_cb
        return cb
    
    def train(self, train_data, device, args):
        model = self.model
        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)
        
        epoch_loss = []
        # 初始化梯度累积的存储
        self.grad_accum = []
        # 变，与fedavg同，与qsgd不同
        for param in self.model.parameters():
            self.grad_accum.append(torch.zeros_like(param.data, device=device))

        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

        # 变，与fedavg同，与qsgd不同
                for i in range(len(list(self.model.parameters()))):
                    self.grad_accum[i].add_(list(self.model.parameters())[i].grad.data)
            
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())

        # 变，与fedavg同，与qsgd不同

            epoch_loss.append(sum(batch_loss) / len(batch_loss))

            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))  



    def test(self, test_data, device, args):
        model = self.model

        model.to(device)
        model.eval()

        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
                x = x.to(device)
                target = target.to(device)
                pred = model(x)
                loss = criterion(pred, target)

                _, predicted = torch.max(pred, -1)
                correct = predicted.eq(target).sum()

                metrics['test_correct'] += correct.item()
                metrics['test_loss'] += loss.item() * target.size(0)
                metrics['test_total'] += target.size(0)
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False

