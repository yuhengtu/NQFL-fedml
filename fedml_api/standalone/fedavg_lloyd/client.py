import logging
import numpy as np

#变0
from fedml_api.standalone.fedavg_lloyd.lloyd_quant import *


class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args
        self.device = device
        self.model_trainer = model_trainer


    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        self.client_idx = client_idx
        self.local_training_data = local_training_data
        self.local_test_data = local_test_data
        self.local_sample_number = local_sample_number

# 变1
    def lloyd_quantize(self, gradients, q):
        q_gradients = []
        print(f'-----------------gradients[0].shape:{gradients[0].shape}-----------------------------')
        # np.savetxt('gradients_0.txt', gradients[0].cpu().numpy())
        # 权重的梯度
        # [10， 784]，（lr模型，28^2 = 784个参数）
        # 输入特征的维度为 n，输出特征的维度为 m，则该层的参数矩阵的形状为 (m,n)
        # 10个输出神经元，每行一个，全连接到784个输入神经元

        print(f'-----------------gradients[1]:{gradients[1]}-----------------------------')
        print(f'-----------------gradients[1].shape:{gradients[1].shape}-----------------------------')
        # bias的梯度
        # [10]

        # w * x + b
        # [10, 784] * [784, 1] + [10]

        for i in range(len(gradients)):
            quantized_g = quantize(gradients[i], {'n': q})
            q_gradients.append(quantized_g)
        print(f'-----------------q_gradients[0].shape:{q_gradients[0].shape}-----------------------------')
        # np.savetxt('q_gradients_0.txt', q_gradients[0].cpu().numpy())
        print(f'-----------------q_gradients[1]:{q_gradients[1]}-----------------------------')
        print(f'-----------------q_gradients[1].shape:{q_gradients[1].shape}-----------------------------')

        return q_gradients

    

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        # weights = self.model_trainer.get_model_params()
        gradients = self.model_trainer.get_model_gradients()
        # 变2
        q_gradients = self.lloyd_quantize(gradients, self.args.quantized_bits)
        communication_bits = self.model_trainer.get_comm_bits(self.args.quantized_bits)
        return q_gradients, communication_bits

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics