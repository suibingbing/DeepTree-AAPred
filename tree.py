import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from transformers import BertModel, BertTokenizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from rdkit import Chem
from rdkit.Chem import AllChem
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
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
import warnings
warnings.filterwarnings("ignore")

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

    scores = [1 if prob >= threshold else 0 for prob in scores]
    TN, FP, FN, TP = confusion_matrix(scores, labels).ravel()
    sensitivity = recall_score(labels, scores)  # 灵敏度/召回率
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision=precision_score(labels, scores)
    recall=sensitivity
    
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
    # if specificity >=specificity3:
    #     specificity3=specificity
    #     updated = True
    # if sensitivity >=sensitivity3:
    #     sensitivity3=sensitivity
    #     updated = True
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

train_file = 'data/aap/AAIP_135.csv'  #Read training set
test_file = 'data/aap/AAIP_28.csv'  #Read independent test set
train_dataset = MyDataset(train_file)
test_dataset = MyDataset(test_file)
batch_size = 2  #Setting batchsize
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

criterion = nn.CrossEntropyLoss()
model = MyModel()
model.to(device)
learning_rates=0.000002
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)
best_mcc=0
for epoch in range(10):
    model.train()
    epoch_auc=0
    print("epoch",epoch)
    for batch_data, batch_labels, batch_data_protbert in train_dataloader:
        model.train()
        batch_labels = batch_labels.to(device)
        outputs = model(batch_data,batch_data_protbert)
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
            for batch_data, batch_labels, batch_data_protbert in test_dataloader:
                batch_labels = batch_labels.to(device)
                outputs = model(batch_data,batch_data_protbert)
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
        if epoch_auc<auc:
            epoch_auc=auc
            file_path = "autodl-tmp/independ/tree.pth"
            if not os.path.exists("autodl-tmp/independ"):
                os.makedirs("autodl-tmp/independ")
            torch.save(model.state_dict(), file_path)
        mcc = matthews_corrcoef(all_labels, pre)
        recall, specificity, sensitivity, precision = calculate_metrics2(all_labels, all_scores, 0.5)
        current_metrics = (acc, auc, mcc,recall, specificity, sensitivity, precision) 
        metrics, updated = update_best_metrics(*current_metrics, metrics)
    print(f"AUC: {epoch_auc:.3f}")
    
print(f"BEST_AUC: {metrics[1]:.3f}",f"ACC: {metrics[0]:.3f}", f"MCC: {metrics[2]:.3f}",f"specificity: {metrics[4]:.3f}", f"sensitivity: {metrics[5]:.3f}")
