#import package
import torch
import warnings
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
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
from rdkit import RDLogger

# 禁用 RDKit 的日志输出
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)
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

warnings.filterwarnings("ignore")
def finger(sequence):
    mol = Chem.MolFromFASTA(sequence)
    fp = np.zeros(128)
    if mol:
        fps_nonzero = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=128)
        for i in range(len(fp)):
                  fp[fps_nonzero[i]] = 1
    else:
        print(sequence)
    return fp

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
import torch
import torch.nn.functional as F

def pad_tensor_to_shape(tensor, target_shape):
    """
    将给定 tensor 补零为目标形状。

    参数:
    tensor (torch.Tensor): 需要补零的 tensor。
    target_shape (tuple): 目标形状 (batch_size, target_dim, width)。

    返回:
    torch.Tensor: 补零后的 tensor。
    """
    current_shape = tensor.shape
    assert len(current_shape) == len(target_shape), "输入的 tensor 和目标形状必须具有相同的维度数量"
    
    # 计算第二维度需要补零的数量
    padding_needed = target_shape[1] - current_shape[1]
    
    assert padding_needed >= 0, "目标形状的第二维度必须大于等于当前形状的第二维度"
    
    # 使用 F.pad 函数补零
    padded_tensor = F.pad(tensor, (0, 0, 0, padding_needed))
    
    return padded_tensor



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
        fp=finger(self.sequence[index])
        return sample, label, sample_protbert,fp
    
    
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
        # outputs_esm_last=pad_tensor_to_shape(outputs_esm_last,(2, 15, 480))

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
        # outputs_pro_last=pad_tensor_to_shape(outputs_pro_last,(2, 15, 1024))
        
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
        # self.fc_class2 = nn.Linear(1024, 480)
        # self.fc_class3 = nn.Linear(480, 256)
        # self.fc_class4 = nn.Linear(256, 2)
        # self.fc_class5 = nn.Linear(128, 64)
        # self.fc_class6 = nn.Linear(64, 32)
        # self.fc_class7 = nn.Linear(32, 16)
        # self.fc_class8 = nn.Linear(16, 8)
        # self.fc_class9 = nn.Linear(8, 4)
        # self.fc_class10 = nn.Linear(4, 2)
        
    def forward(self, inputs,inputs2,fp):
        
        output_cnn_esm,lstm_output_esm=self.esm_based(inputs)
        output_cnn_pro,lstm_output_pro=self.prot_based(inputs2)
        fp=fp.to(device)
        x=output_cnn_pro+output_cnn_esm+lstm_output_esm+lstm_output_pro+fp
        # x=self.fc_class1(torch.cat((output_cnn_pro,output_cnn_esm,lstm_output_esm,lstm_output_pro),dim=1))
        # x=self.sigmoid(x)
        # x=self.fc_class2(x)+outputs_esm_pool
        # x=self.sigmoid(x)
        # x=self.fc_class3(x)
        # x=self.sigmoid(x)
        # x=self.fc_class4(x)
        # x=self.sigmoid(x)
        # x=self.fc_class5(x)
        # x=self.sigmoid(x)
        # x=self.fc_class6(x)
        # x=self.sigmoid(x)
        # x=self.fc_class7(x)
        # x=self.sigmoid(x)
        # x=self.fc_class8(x)
        # x=self.sigmoid(x)
        # x=self.fc_class9(x)
        # x=self.sigmoid(x)
        # x=self.fc_class10(x)
        return x
#Read the dataset

train_file = 'AAIP_135.csv'  #Read training set
test_file = 'AAIP_28.csv'  #Read independent test set
train_dataset = MyDataset(train_file)
test_dataset = MyDataset(test_file)
batch_size = 2  #Setting batchsize
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
#Model loading and setting

criterion = nn.CrossEntropyLoss()
model = MyModel()#Model loading
model.to(device)
learning_rates=0.000001 #Setting learning rates
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)
# optimizer = optim.Adam([
#     {'params': model.esm_based.parameters(), 'lr': 0.0000661},
#     {'params': model.prot_based.parameters(), 'lr': 0.00001},
#     {'params': model.fc_class1.parameters(), 'lr': 1e-3},
# ])
#Model training and evaluation
best_mcc=0
for epoch in range(10):
    item=0
    model.train()
    print("epoch",epoch)
    for batch_data, batch_labels, batch_data_protbert,fp in train_dataloader:
        item=item+1
        model.train()
        batch_labels = batch_labels.to(device)
        outputs = model(batch_data,batch_data_protbert,fp)
        loss = criterion(outputs, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    all_labels = []
    all_scores = []
    pre=[]
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_predictions = 0
        for batch_data, batch_labels, batch_data_protbert,fp in test_dataloader:
            batch_labels = batch_labels.to(device)
            outputs = model(batch_data,batch_data_protbert,fp)
            probabilities = nn.functional.softmax(outputs, dim=1)
            scores = probabilities[:, 1]  
            all_labels.extend(batch_labels.tolist())
            all_scores.extend(scores.tolist())
            predicted_labels = scores >= 0.5 
            pre.extend(predicted_labels.tolist())
            correct_predictions += (predicted_labels == batch_labels).sum().item()
            total_predictions += batch_labels.size(0)
    acc = correct_predictions / total_predictions
    auc = roc_auc_score(all_labels, all_scores)
    mcc = matthews_corrcoef(all_labels, pre)
    recall, specificity, sensitivity, precision = calculate_metrics2(all_labels, all_scores, 0.5)
    current_metrics = (acc, auc, mcc,recall, specificity, sensitivity, precision) 
    metrics, updated = update_best_metrics(*current_metrics, metrics)
    print(auc,acc,mcc)
print(f"BEST_AUC: {metrics[1]:.3f}",f"ACC: {metrics[0]:.3f}", f"MCC: {metrics[2]:.3f}",f"specificity: {metrics[4]:.3f}", f"sensitivity: {metrics[5]:.3f}")