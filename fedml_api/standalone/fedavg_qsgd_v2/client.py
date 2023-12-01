import logging

# 变2
from fedml_api.standalone.fedavg_qsgd.qsgd import *


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

    # 变3
    def qsgd_quantize(self, gradients, q):
        q_gradients = []
        for i in range(len(gradients)):
            quantized_g = quantize(gradients[i], {'n': q})
            q_gradients.append(quantized_g)
        return q_gradients

    

    def get_sample_number(self):
        return self.local_sample_number

    def train(self, w_global):
        self.model_trainer.set_id(self.client_idx)
        self.model_trainer.set_model_params(w_global)
        self.model_trainer.train(self.local_training_data, self.device, self.args)
        # weights = self.model_trainer.get_model_params()
        gradients = self.model_trainer.get_model_gradients()
        # 变4
        q_gradients = self.qsgd_quantize(gradients, self.args.quantized_bits)
        communication_bits = self.model_trainer.get_comm_bits(self.args.quantized_bits)
        return q_gradients, communication_bits

    def local_test(self, b_use_test_dataset):
        if b_use_test_dataset:
            test_data = self.local_test_data
        else:
            test_data = self.local_training_data
        metrics = self.model_trainer.test(test_data, self.device, self.args)
        return metrics
        