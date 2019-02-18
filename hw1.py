
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset

from collections import defaultdict

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEVICE


# In[ ]:


import io

def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = map(float, tokens[1:])
    return data

w2v = load_vectors("./wiki-news-300d-1M.vec")
for key in w2v.keys():
    w2v[key] = np.array(list(w2v[key]))
    
truncated_vocab = np.load("vocab.npy").item()


# In[ ]:


t2i = defaultdict(lambda: len(t2i))

def read_dataset(filename):
    embed_data = []
    label = []
    with open(filename, "r") as f:
        for line in f:
            topic, words = line.lower().strip().split(" ||| ")
            words = words.split(" ")
            embed_line = []
            for w in words:
                if w in truncated_vocab and w in w2v:
                    embed_line.append(np.array(w2v[w]))
                else:
                    embed_line.append(np.random.random(300))
            embed_data.append(np.array(embed_line))
            label.append(t2i[topic])
    return embed_data, label

train_data, train_labels = read_dataset("./topicclass/topicclass_train.txt")
test_data, test_labels = read_dataset("./topicclass/topicclass_valid_corrected.txt")
true_test_data, _ = read_dataset("./topicclass/topicclass_test.txt")


# In[ ]:


del w2v
del truncated_vocab


# In[ ]:


import random

label2idx = {}
for i, label in enumerate(train_labels):
    if label in label2idx:
        label2idx[label].append(i)
    else:
        label2idx[label] = [i]
        
train_idx = []
val_idx = []
for topic in label2idx:
    idx = label2idx[topic]
    random.shuffle(idx)
    sep = int(0.95 * len(idx))
    train_idx.extend(idx[:sep])
    val_idx.extend(idx[sep:])
    
val_data = [train_data[i] for i in val_idx]
val_labels = [train_labels[i] for i in val_idx]
train_data = [train_data[i] for i in train_idx]
train_labels = [train_labels[i] for i in train_idx]


# In[ ]:


max_seq_len = max([len(seq) for seq in train_data])
for d in range(len(train_data)):
    pad_len = max_seq_len - len(train_data[d])
    train_data[d] = np.append(train_data[d], np.zeros((pad_len,300)), axis=0)

train_data = torch.FloatTensor(train_data)
train_labels = torch.LongTensor(train_labels)
print ("DONE for training dataset")


# In[ ]:


max_seq_len = max([len(seq) for seq in val_data])
for d in range(len(val_data)):
    pad_len = max_seq_len - len(val_data[d])
    val_data[d] = np.append(val_data[d], np.zeros((pad_len,300)), axis=0)

val_data = torch.FloatTensor(val_data)
val_labels = torch.LongTensor(val_labels)
print ("DONE for validation dataset")


# In[ ]:


max_seq_len = max([len(seq) for seq in test_data])
for d in range(len(test_data)):
    pad_len = max_seq_len - len(test_data[d])
    test_data[d] = np.append(test_data[d], np.zeros((pad_len,300)), axis=0)

test_data = torch.FloatTensor(test_data)
test_labels = torch.LongTensor(test_labels)
print ("DONE for testing dataset")


# In[ ]:


# Dataset
class HW1Dataset(Dataset):
    def __init__(self, data, label, has_label=True):
        self.data = data
        self.label = label
        self.has_label = has_label
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, i):
        if self.has_label:
            return self.data[i], self.label[i]
        else:
            return self.data[i]

train_dataset = HW1Dataset(train_data, train_labels)
val_dataset = HW1Dataset(val_data, val_labels)
test_dataset = HW1Dataset(test_data, test_labels)


# In[ ]:


class ConvPool(nn.Module):
    def __init__(self, embed_size, feat_size, kernel_size):
        super(ConvPool,self).__init__()
        
        self.conv = nn.Conv1d(in_channels=embed_size, out_channels=feat_size, 
                              kernel_size=kernel_size)
#         self.bn = nn.BatchNorm1d(feat_size)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        nn.init.xavier_normal_(self.conv.weight)
#         nn.init.constant_(self.bn.weight, 1)
#         nn.init.constant_(self.bn.bias, 0)
        
    def forward(self, x):
        x_conv = self.conv(x)
        out = self.pool(self.relu(x_conv))
        return out

class Net(nn.Module):
    def __init__(self, embed_size, feat_size, kernel_sizes, ntopics, nwords=0):
        super(Net,self).__init__()
        
        #self.embed = nn.Embedding(nwords, embed_size) # batch_size * L * embed_size
        self.convs = nn.ModuleList([ConvPool(embed_size, feat_size, i) for i in kernel_sizes])
        self.dropout = nn.Dropout(p=0.8)
        self.fc1 = nn.Linear(len(kernel_sizes) * feat_size, ntopics)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(feat_size, ntopics)
        
        nn.init.xavier_normal_(self.fc1.weight)
#         nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, x):
        #x_embed = self.embed(x).transpose(1,2)
        x_embed = x.transpose(1,2)
        x_conv = []
        for conv in self.convs:
            x_conv.append(conv(x_embed))
        x_conv = torch.cat(x_conv, dim=1).squeeze()       
        out = self.fc1(self.dropout(x_conv))
#         out = self.fc2(self.relu(out))
        return out


# In[ ]:


embed_size = 300
feat_size = 128
kernel_sizes = [3,4,5]
l2_const = 3

num_epoch = 10
batch_size = 50


# In[ ]:


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size,shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)


# In[ ]:


model = Net(embed_size, feat_size, kernel_sizes, 16).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

print (model)


# In[ ]:


for i in range(num_epoch):
    model.train()
    #scheduler.step()
    
    train_loss = 0.0
    train_acc = 0
    num_batch = 0
    for data, label in train_loader:
        data = data.to(DEVICE)
        label = label.to(DEVICE)
        num_batch += 1
        
        scores = model(data)
        #if train_loader.batch_size == 1:
        #    scores = scores[np.newaxis,:]
        loss = criterion(scores, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # l2 norm constraint
        w1 = model.fc1.weight.data
        if w1.norm(p=2).item() > l2_const:
            w1.div_(w1.norm(p=2) / l2_const)
#         w2 = model.fc2.weight.data
#         if w2.norm(p=2).item() > l2_const:
#             w2.div_(w2.norm(p=2) / l2_const)
        
        train_loss += loss.item()
        prediction = torch.argmax(scores.data, dim=1)
        train_acc += torch.sum(prediction.eq(label)).item()

    train_loss /= num_batch
    train_acc = 1.0 * train_acc / len(train_data)
    
    print ('[TRAIN] Epoch [%d/%d]    Loss: %.4f    Acc: %.4f' % 
            (i+1,num_epoch,train_loss,train_acc))
    
    model.eval()
    val_acc = 0
    val_loss = 0.0
    num_batch = 0
    for v_data, v_label in val_loader:
        v_data = v_data.to(DEVICE)
        v_label = v_label.to(DEVICE)
        num_batch += 1
        
        v_scores = model(v_data)
        if v_label.shape == torch.Size([1]):
            v_scores = v_scores[np.newaxis,:]
        
        loss = criterion(v_scores, v_label)        
        val_loss += loss.item()
        prediction = torch.argmax(v_scores.data, dim=1)
        val_acc += torch.sum(prediction.eq(v_label)).item()
    
    val_loss /= num_batch
    val_acc = 1.0 * val_acc / len(val_data)
    
    print ('[VAL] Epoch [%d/%d]    Loss: %.4f    Acc: %.4f' % 
            (i+1,num_epoch,val_loss,val_acc))


# In[ ]:


test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

model.eval()
test_acc = 0
test_loss = 0.0
num_batch = 0
for t_data, t_label in test_loader:
    t_data = t_data.to(DEVICE)
    t_label = t_label.to(DEVICE)
    num_batch += 1
        
    t_scores = model(t_data)
    if t_label.shape == torch.Size([1]):
        t_scores = t_scores[np.newaxis,:]
        
    loss = criterion(t_scores, t_label)        
    test_loss += loss.item()
    prediction = torch.argmax(t_scores.data, dim=1)
    test_acc += torch.sum(prediction.eq(t_label)).item()
    
test_loss /= num_batch
test_acc = 1.0 * test_acc / len(test_data)
    
print ('[VAL] Epoch [%d/%d]    Loss: %.4f    Acc: %.4f' % 
        (i+1,num_epoch,test_loss,test_acc))


# In[ ]:


model.load_state_dict(torch.load("./128feat_0.78.pt"))


# In[ ]:


true_test_dataset = HW1Dataset(true_test_data, None, False)
true_test_loader = DataLoader(dataset=true_test_data, batch_size=1, shuffle=False)

model.eval()
predictions = []
for tt_data in true_test_loader:
    tt_data = tt_data.float().to(DEVICE)
    
    tt_scores = model(tt_data)[np.newaxis,:]
    pred = torch.argmax(tt_scores.data,dim=1)
    predictions.append(pred.item())

with open("prediction.txt", "w") as f:
    for pred in predictions:
        for topic, idx in t2i.items():
            if idx == pred:
                f.write(topic + "\n")
                break
print ("finish prediction")

