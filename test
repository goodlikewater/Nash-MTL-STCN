import torch.nn as nn
from torch.nn.utils import weight_norm
import torch


#################  1-D TCN Model by Mustafa et al. ###################################################
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        """因果卷积"""
        # self.chomp1 = Chomp1d(padding)
 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True))
        
        """因果卷积"""
        # self.chomp2 = Chomp1d(padding)

        
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        #自注意力层        
        # self.encoder_att = nn.MultiheadAttention(embed_dim=n_outputs, num_heads=1, dropout=dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                  self.conv2, self.relu2, self.dropout2)
        #下采样
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:#归一化层参数化网络的权重,加快收敛
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        # print("out.shape", out.shape)
        res = x if self.downsample is None else self.downsample(x) 
        
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        # print('num_levels:',num_levels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=int((kernel_size - 1)/2 * dilation_size), dropout=dropout)]

        self.network = nn.Sequential(*layers)#将多个层组合成一个复杂的神经网络模型;"*"一个列表中的元素逐个赋值给变量

    def forward(self, x):
        return self.network(x)

 
class MustafaNet(nn.Module):
    def __init__(self):
        super(MustafaNet, self).__init__()
        self.tcn_local = TemporalConvNet(num_inputs=6, num_channels=[90, 180, 180, 90], kernel_size=9, dropout=0.2)     
        self.regression = RegressionModule()
        self.task1 = task1_Module()
        self.task2 = task2_Module()
        self.task3 = task3_Module()

    
    # def shared_parameters(self):
    #     return [p for n,p in self.named_parameters() if 'tcn_local' in n] #tcn_local or regression
    # def task_specific_parameters(self):
    #     return [p for n,p in self.named_parameters() if 'regression' in n]

    def shared_parameters(self):
        return [p for n, p in self.named_parameters() if not any(task_name in n for task_name in ['task1', 'task2', 'task3'])]
    def task_specific_parameters(self):
        return [p for n, p in self.named_parameters() if any(task_name in n for task_name in ['task1', 'task2', 'task3'])]
    def last_shared_parameters(self):
        last_layer_params = list(self.regression.conv4.parameters())
        return last_layer_params


    def forward(self, input):
        out = self.tcn_local(input)
        out = self.regression(out)
        task1_output = self.task1(out)
        task2_output = self.task2(out)
        task3_output = self.task3(out)
        task1_output = task1_output.squeeze(1)
        task2_output = task2_output.squeeze(1)
        task3_output = task3_output.squeeze(1)
    
        output = torch.stack([task1_output, task2_output, task3_output],dim=1)   
        return output        


class task1_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 1, kernel_size=1)        
    def forward(self, x):
        out = self.conv1(x)
        return out
    
class task2_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 1, kernel_size=1)  
    def forward(self, x):
        out = self.conv1(x)
        return out
    
class task3_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(64, 1, kernel_size=1)
    def forward(self, x):
        out = self.conv1(x)
        return out
    
    
class RegressionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(90, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )  
        
        self.conv4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )   

        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)   
        out = self.conv3(out)
        out = self.conv4(out)

        return out


