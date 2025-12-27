import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchsummary import summary
import time


# 提高准确率方法:
# 1.优化方法从SGD->Adam
# 2.学习率从0.001->0.0001
# 3.对数据进行标准化
# 4.增加网络深度，每层的神经元数量
# 5.增加训练轮数

#1. 创建数据集
def create_dataset():
    #加载数据集
    data = pd.read_csv('data/手机价格预测.csv')#（2000，21）
    # print(f'data: {data.head()}')
    # print(f'data: {data.shape}')

    #获取x特征列和y标签列
    x,y = data.iloc[:,:-1],data.iloc[:,-1]#x大小为(2000,20),y大小为(2000,1)
    # print(f'x:{x.head()},{x.shape}')
    # print(f'y:{y.head()},{y.shape}')

    #将x转成浮点型
    x = x.astype(np.float32)

    #切分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(
        x,y,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )#x_train大小为（1600，20），x_test大小为（400，20），y_train大小为（1600，1），y_test大小为（400，1），
    # print(f'x:{x_train.head()},{x_train.shape}')
    # print(f'y:{y_train.head()},{y_train.shape}')

    #把数据集封装成张量数据集 思路：数据->张量tensor->数据集TensorDataset->数据加载器Dataloader
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))
    # print(f':train_dataset:{train_dataset.data}')
    # print(f':test_dataset:{test_dataset}')

    # print(x.shape[1], len(np.unique(y)))

    return train_dataset, test_dataset, x.shape[1], len(np.unique(y))

#2. 构建神经网络
class Model(nn.Module):
    def __init__(self, input_dim, output_dim):
        #初始化父类成员
        super().__init__()
        #搭建神经网络
        # 隐藏层1
        self.linear1 = nn.Linear(input_dim, 128)
        # 隐藏层2
        self.linear2 = nn.Linear(128, 256)
        # 隐藏层3
        self.linear3 = nn.Linear(256, 512)
        # 隐藏层4
        self.linear4 = nn.Linear(512, 256)
        # 隐藏层5
        self.linear5 = nn.Linear(256, 128)
        # 输出层
        self.output = nn.Linear(128, output_dim)

    #定义前向传播方法
    def forward(self, x):
        # 隐藏层1：加权求和+relu激活
        x = torch.relu(self.linear1(x))
        # 隐藏层2：加权求和+relu激活
        x = torch.relu(self.linear2(x))
        # 隐藏层3：加权求和+relu激活
        x = torch.relu(self.linear3(x))
        # 隐藏层4：加权求和+relu激活
        x = torch.relu(self.linear4(x))
        # 隐藏层5：加权求和+relu激活
        x = torch.relu(self.linear5(x))
        # 输出层：加权求和
        x = self.output(x)

        return x

#3. 模型训练
def train(train_dataset, input_dim, output_dim):
    #创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    #创建网络模型
    model = Model(input_dim, output_dim)
    #定义损失函数
    criterion = nn.CrossEntropyLoss()
    #创建优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #定义训练轮数
    epochs = 500
    #开启多轮训练
    for epoch in range(epochs):
        #定义每次训练的损失值，训练批次数
        total_loss, batch_num = 0.0, 0
        #定义训练开始时间
        start = time.time()
        #开启一轮训练
        for x, y in train_loader:#x大小为(16,20)，y大小为(16,1)
            #切换到训练模式
            model.train()
            #模型预测
            y_pred = model(x)#y_pred大小为(16,4)
            #计算损失
            loss = criterion(y_pred, y)
            #梯度清零，方向传播，优化参数
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1
        #打印训练信息
        print(f'epoch: {epoch+1}, loss: {total_loss/batch_num:.4f}, time: {time.time() - start:.2f}s')
    #训练结束，保存模型
    torch.save(model.state_dict(), 'model/model.pth')

#4. 模型测试
def evaluate(test_dataset, input_dim, output_dim):
    #创建数据加载器
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    #创建网络模型
    model = Model(input_dim, output_dim)# input_dim=20,output_dim=4
    #加载模型参数
    model.load_state_dict(torch.load('model/model.pth'))
    #定义变量
    correct = 0
    for x, y in test_loader:#x大小为(8,20)，y大小为(8,1)
        #切换到测试模型
        model.eval()
        #模型预测
        y_pred = model(x)#y_pred大小为(8,4),[[0分类概率，1分类概率，2分类概率，3分类概率],[...],...]
        #利用argmax()获取8行中每一行最大值的索引
        y_pred = torch.argmax(y_pred, dim=1)#y_pred大小为(8,1),[第1行预测分类，第2行预测分类，...]
        # print(f'y_pred: {y_pred}')
        #统计预测正确的样本数
        correct += (y_pred == y).sum()
    #打印准确率
    print(f'准确率: {correct/len(test_dataset):.4f}')


if __name__ == '__main__':
    #准备数据集
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()# input_dim=20, output_dim=4
    #模型训练
    # train(train_dataset, input_dim, output_dim)
    #模型预测
    evaluate(test_dataset, input_dim, output_dim)
