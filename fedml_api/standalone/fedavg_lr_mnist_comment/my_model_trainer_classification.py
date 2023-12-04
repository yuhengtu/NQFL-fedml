import logging

import torch
from torch import nn
from scipy.stats import norm
import numpy as np

from fedml_core.trainer.model_trainer import ModelTrainer

class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()
        # 返回模型的参数字典，键是参数的名称，值是对应参数的张量。
    
    def get_model_params_cuda(self, device):
        return self.model.to(device).state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)
    
    def get_model_gradients(self):
        return self.grad_accum
    
    def get_comm_bits(self):
        vars = self.grad_total.var(0) # 梯度的方差，维度0
        cb = 0.5 * torch.abs(torch.log2(vars[vars!=0])).sum().item() + 0.5 * np.log2(2 * np.pi * np.e) * vars.shape[0]
        # .item() 只包含一个元素的tensor转换为数值类型
        return cb
    

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        
        epoch_loss = []
        # grad_accum，存储每个模型参数对应的，每个 batch 梯度的累积值
        self.grad_accum = []

        # grad_epoch，存储每个模型参数对应的，每轮 epoch 的梯度
        grad_epoch = []
        parameters = self.get_model_params_cuda(device)
        for k, v in parameters.items(): #key，value
            # 为每个模型参数创建两个空张量
            self.grad_accum.append(torch.zeros_like(v.data, device=device))
            grad_epoch.append(torch.zeros_like(v.data, device=device))
        
        # 所有轮的梯度
        self.grad_total = None
        
        for epoch in range(args.epochs):
            batch_loss = []
            for t in grad_epoch:
                t.zero_()
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # 梯度累积
                for i, k in enumerate(parameters.keys()):
                    # self.grad_accum[i].add_(list(self.model.parameters())[i].grad.data)
                    if parameters[k].grad != None:
                        grad_epoch[i].add_(parameters[k].grad.data)
            
                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            
            for i, k in enumerate(parameters.keys()):
                if parameters[k].grad != None:
                    grad_epoch[i].div_(len(train_data))
                    self.grad_accum[i].add_(grad_epoch[i].data)
            
            grad_cur = grad_epoch[0].reshape(-1)
            for i in range(1, len(parameters.keys())):
                grad_cur = torch.cat((grad_cur, grad_epoch[i].reshape(-1)))
            if self.grad_total is None:
                self.grad_total = grad_cur.unsqueeze(0)
            else:
                self.grad_total = torch.cat((self.grad_total, grad_cur.unsqueeze(0)))

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


