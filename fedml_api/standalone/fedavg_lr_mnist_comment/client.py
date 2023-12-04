import logging

class Client:

    def __init__(self, client_idx, local_training_data, local_test_data, local_sample_number, args, device,
                 model_trainer):
        # 初始化客户端对象
        self.client_idx = client_idx  # 客户端索引
        self.local_training_data = local_training_data  # 本地训练数据
        self.local_test_data = local_test_data  # 本地测试数据
        self.local_sample_number = local_sample_number  # 本地样本数量
        logging.info("self.local_sample_number = " + str(self.local_sample_number))

        self.args = args  # 参数
        self.device = device  # 设备（例如CPU或GPU）
        self.model_trainer = model_trainer  # 模型训练器

    def update_local_dataset(self, client_idx, local_training_data, local_test_data, local_sample_number):
        # 更新本地数据集
        self.client_idx = client_idx  # 更新客户端索引
        self.local_training_data = local_training_data  # 更新本地训练数据
        self.local_test_data = local_test_data  # 更新本地测试数据
        self.local_sample_number = local_sample_number  # 更新本地样本数量

    def get_sample_number(self):
        # 获取本地样本数量
        return self.local_sample_number

    def train(self, w_global):
        # 客户端训练方法
        self.model_trainer.set_id(self.client_idx)  # 设置模型训练器的客户端ID
        self.model_trainer.set_model_params(w_global)  # 设置模型参数为全局参数
        self.model_trainer.train(self.local_training_data, self.device, self.args)  # 使用本地数据集进行模型训练
        # weights = self.model_trainer.get_model_params()
        gradients = self.model_trainer.get_model_gradients()  # 获取模型梯度
        communication_bits = self.model_trainer.get_comm_bits()  # 获取通信比特数
        return gradients, communication_bits

    def local_test(self, b_use_test_dataset):
        # 客户端本地测试方法
        if b_use_test_dataset:
            test_data = self.local_test_data  # 使用测试数据集
        else:
            test_data = self.local_training_data  # 使用训练数据集
        metrics = self.model_trainer.test(test_data, self.device, self.args)  # 在本地数据集上进行模型测试
        return metrics  # 返回测试指标
