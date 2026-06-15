import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
import matplotlib.pyplot as plt
from rdkit import Chem
from sklearn.metrics import roc_auc_score,roc_curve
from rdkit.Chem import AllChem
from scipy.interpolate import interp1d
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import matthews_corrcoef
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import torch.optim as optim
from sklearn.model_selection import train_test_split
import random
import numpy as np
import torch
import csv
#Calculation of indicators and dataset definitions

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random_seed = 42
loss_all=99999
metrics = (0, 0, 0,0,0,0,0)  
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def calculate_metrics2(labels, scores, threshold):
    
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    binary_predictions = [1 if scores[i] >= threshold else 0 for i in sorted_indices]
    TP = sum([1 for i in range(len(labels)) if labels[i] == 1 and binary_predictions[i] == 1])
    FP = sum([1 for i in range(len(labels)) if labels[i] == 0 and binary_predictions[i] == 1])
    TN = sum([1 for i in range(len(labels)) if labels[i] == 0 and binary_predictions[i] == 0])
    FN = sum([1 for i in range(len(labels)) if labels[i] == 1 and binary_predictions[i] == 0])
    recall = TP / (TP + FN) if TP + FN > 0 else 0.0
    specificity = TN / (TN + FP) if TN + FP > 0 else 0.0
    sensitivity = recall
    precision = TP / (TP + FP) if TP + FP > 0 else 0.0
    return recall, specificity, sensitivity, precision

    

def update_best_metrics(a, b, c, recall, specificity, sensitivity, precision, metrics):
    d, e, f,recall3, specificity3, sensitivity3, precision3= metrics
    updated = False 
    if a >=d:
        d = a
        recall3=recall
        specificity3=specificity
        sensitivity3=sensitivity
        precision3=precision
        updated = True
    if specificity >=specificity3:
        specificity3=specificity
        updated = True
    if sensitivity >=sensitivity3:
        sensitivity3=sensitivity
        updated = True
    if b >= e:
        e = b
        updated = True
    if f >= c:
        c = f
        updated = True
    return (d, e, c,recall3, specificity3, sensitivity3, precision3), updated

def process_sequence(sequence):
    max_length = 15
    if len(sequence) > max_length:
        return sequence[:max_length]
    else:
        return sequence + '0' * (max_length - len(sequence))

class MyDataset(Dataset):
    def __init__(self, file):
        self.sequence, self.label = self.read_file(file)
        self.sequence_protbert=self.add_space_between_characters(self.sequence)
        
                # 将数据打乱
        combined = list(zip(self.sequence, self.label, self.sequence_protbert))
        random.shuffle(combined)
        self.sequence, self.label, self.sequence_protbert = zip(*combined)
        
    def read_file(self,file_path):
        sequences = []
        labels = []
        with open(file_path, 'r', newline='') as csv_file:
            csv_reader = csv.reader(csv_file)
            next(csv_reader, None) 
            for row in csv_reader:
                sequences.append(row[0])
                labels.append(row[1])
        return sequences, labels
    
    def add_space_between_characters(self,input_list):
        new_list = []
        for element in input_list:
            new_element = ' '.join(element)
            new_list.append(new_element)
        return new_list

    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, index):
        
        sample=sequence = process_sequence(self.sequence[index])
        sample_protbert=self.sequence_protbert[index]
        label=int(self.label[index])
        return sample, label, sample_protbert
    
    
#FusPB-ESM2 model Definition
class ESM_Model(nn.Module):
    def __init__(self,):
        super(ESM_Model, self).__init__()
        
        self.model = AutoModel.from_pretrained("ESM")
        self.tokenizer = AutoTokenizer.from_pretrained("ESM")
        self.bilstm1 = nn.LSTM(self.model.config.hidden_size, 64, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc_esm = nn.Linear(99, 128)  
        self.conv1=nn.Conv2d(1,3,kernel_size=(4,480))
        self.conv2=nn.Conv2d(1,3,kernel_size=(5,480))
        self.conv3=nn.Conv2d(1,3,kernel_size=(6,480))
        
    def forward(self, inputs):
        inputs = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt",max_length=15)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        outputs_esm = self.model(input_ids=input_ids, attention_mask=attention_mask)
        outputs_esm_pool = outputs_esm.pooler_output   
        outputs_esm_last = outputs_esm.last_hidden_state  
        batch_esm=outputs_esm_last.size(0)
        outputs_esm_last=outputs_esm_last.unsqueeze(1)
        output1_esm=self.conv1(outputs_esm_last).view(batch_esm,-1)
        output2_esm=self.conv2(outputs_esm_last).view(batch_esm,-1)
        output3_esm=self.conv3(outputs_esm_last).view(batch_esm,-1)
        output_cnn_esm=torch.cat((output1_esm,output2_esm,output3_esm),dim=1)
        output_cnn_esm=self.fc_esm(output_cnn_esm)
        lstm_output_esm, _ = self.bilstm1(outputs_esm_pool.unsqueeze(0))
        lstm_output_esm = self.dropout(lstm_output_esm)
        lstm_output_esm = lstm_output_esm.squeeze(0)
        return output_cnn_esm,lstm_output_esm
        
        
        
class Prot_Model(nn.Module):
    def __init__(self,):
        super(Prot_Model, self).__init__()
        self.tokenizer_pro = BertTokenizer.from_pretrained("protbert", do_lower_case=False)
        self.model_pro = BertModel.from_pretrained("protbert")
        self.bilstm2 = nn.LSTM(self.model_pro.config.hidden_size, 64, num_layers=1, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.fc_pro = nn.Linear(99, 128)  
        self.conv4=nn.Conv2d(1,3,kernel_size=(4,1024))
        self.conv5=nn.Conv2d(1,3,kernel_size=(5,1024))
        self.conv6=nn.Conv2d(1,3,kernel_size=(6,1024))

    def forward(self, inputs2):

        encoded_input = self.tokenizer_pro(inputs2, padding=True, truncation=True,return_tensors='pt',max_length=15).to(device)
        outputs_pro = self.model_pro(**encoded_input)
        outputs_pro_pool = outputs_pro.pooler_output   
        outputs_pro_last = outputs_pro.last_hidden_state  
        batch_pro=outputs_pro_last.size(0)
        outputs_pro_last=outputs_pro_last.unsqueeze(1)
        output1_pro=self.conv4(outputs_pro_last).view(batch_pro,-1)
        output2_pro=self.conv5(outputs_pro_last).view(batch_pro,-1)
        output3_pro=self.conv6(outputs_pro_last).view(batch_pro,-1)
        output_cnn_pro=torch.cat((output1_pro,output2_pro,output3_pro),dim=1)
        output_cnn_pro=self.fc_pro(output_cnn_pro)
        lstm_output_pro, _ = self.bilstm2(outputs_pro_pool.unsqueeze(0))
        lstm_output_pro = self.dropout(lstm_output_pro)
        lstm_output_pro = lstm_output_pro.squeeze(0)
        
        return output_cnn_pro,lstm_output_pro

class MyModel(nn.Module):
    def __init__(self,):
        super(MyModel, self).__init__()
        self.esm_based=ESM_Model()
        self.prot_based=Prot_Model()
        self.sigmoid = nn.Sigmoid()
        self.fc_class1 = nn.Linear(128, 2)
    def forward(self, inputs,inputs2):
        
        output_cnn_esm,lstm_output_esm=self.esm_based(inputs)
        output_cnn_pro,lstm_output_pro=self.prot_based(inputs2)
        x=output_cnn_pro+output_cnn_esm+lstm_output_esm+lstm_output_pro
        return x

train_file = 'AAIP_135.csv'  #Read training set
test_file = 'AAIP_28.csv'  #Read independent test set
train_dataset = MyDataset(train_file)
test_dataset = MyDataset(test_file)
batch_size = 2  #Setting batchsize
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
model = MyModel()
model.to(device)
best_mcc=0
#Model training and evaluation
kf = KFold(n_splits=5, shuffle=False)
best_auc=0
best_acc=0
best_epoch=0
best_epoch2=0
all_fpr = []
all_tpr = []
all_aucs = []
all_accs = []
# 模型训练
for fold, (train_indices, valid_indices) in enumerate(kf.split(train_dataset)):
    # 根据KFold的划分获取训练集和验证集
    best_auc=0
    best_acc=0
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(valid_indices)
    print(train_indices,len(train_indices))
    print(valid_indices,len(valid_indices))
    best_fpr=np.array([])
    best_tpr=np.array([])
    # 创建数据加载器实例，使用SubsetRandomSampler来划分数据
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)
    model = MyModel()#Model loading
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00004)
    item=0
    for epoch in range(10):
        item=item+1
        print(item)
        for batch_data, batch_labels,batch_data_pro in train_dataloader:
            model.train()
            batch_labels = batch_labels.to(device)
            # 前向传播
            outputs = model(batch_data,batch_data_pro)
            loss = criterion(outputs, batch_labels)
            # 反向传播和参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            all_labels = []
            all_scores = []
            model.eval()      
            for batch_data, batch_labels,batch_data_pro in valid_dataloader:
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data,batch_data_pro)
                probabilities = nn.functional.softmax(outputs, dim=1)
                scores = probabilities[:, 1]  # 正类的概率

                # 收集真实标签和预测得分
                all_labels.extend(batch_labels.tolist())
                all_scores.extend(scores.tolist())
            fpr, tpr, _ = roc_curve(all_labels, all_scores)
            # print(len(fpr))
            # print(len(tpr))
            auc = roc_auc_score(all_labels, all_scores)
            correct_predictions = (np.array(all_scores) >= 0.5).astype(int)
            acc = np.mean(correct_predictions == np.array(all_labels))
            if auc>best_auc:
                best_fpr=fpr
                best_tpr=tpr
                best_auc=auc
                file_path = "autodl-tmp/5fold_tree/fold_{}.pth".format(fold)
                if not os.path.exists("autodl-tmp/5fold_tree"):
                    os.makedirs("autodl-tmp/5fold_tree")
                torch.save(model.state_dict(), file_path)
    all_fpr.append(best_fpr)
    all_tpr.append(best_tpr)
    all_aucs.append(best_auc)
    all_accs.append(best_acc)
    print(f"Fold {fold + 1}: AUC = {best_auc:.6f}, Accuracy = {best_acc:.6f}")
plt.figure(figsize=(8, 6))
max_length = max(len(fpr) for fpr in all_fpr)
new_all_fpr = []
new_all_tpr = []
# 进行插值操作
for fpr, tpr in zip(all_fpr, all_tpr):
    f = interp1d(np.linspace(0, 1, len(fpr)), fpr)
    t = interp1d(np.linspace(0, 1, len(tpr)), tpr)
    new_fpr = f(np.linspace(0, 1, max_length))
    new_tpr = t(np.linspace(0, 1, max_length))
    new_all_fpr.append(new_fpr)
    new_all_tpr.append(new_tpr)

all_fpr=new_all_fpr
all_tpr=new_all_tpr

for i in range(len(all_fpr)):
    plt.plot(all_fpr[i], all_tpr[i], linestyle='--',lw=1, label=f'Fold {i + 1} (AUC = {all_aucs[i]:.3f})')

mean_fpr = np.mean(all_fpr, axis=0)
mean_tpr = np.mean(all_tpr, axis=0)

plt.plot(mean_fpr, mean_tpr, color='b', linestyle='-', lw=1.5, label='Mean ROC (AUC = {:.3f})'.format(np.mean(all_aucs)))

# 设置图形属性
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('autodl-tmp/5fold_tree/5_fold_roc_tree.png',dpi=400)

# 显示图形
plt.show()

# 输出每个折叠的AUC和准确率
print("AUC for each fold:", all_aucs)
print("Accuracy for each fold:", all_accs)
print("Mean AUC:", np.mean(all_aucs))
print("Mean Accuracy:", np.mean(all_accs))