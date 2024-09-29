import os
import io
import re
import argparse
from torch.utils.data import DataLoader
import random
import torch
import numpy as np
from transformers import BertTokenizer, BertConfig
from torch.optim import Adam
from tqdm import tqdm
from torch import nn
from transformers import BertModel


parser = argparse.ArgumentParser(description="PyTorch Training Script")
# 添加命令行参数
parser.add_argument("--batch_size", type=int, default=16, help="训练的批次大小")
parser.add_argument("--learning_rate", type=float, default=0.001, help="学习率")
parser.add_argument("--sample_size", type=int, default=1000, help="样本数")
parser.add_argument("--coef", type=float, default=0.02, help="影响因子")
parser.add_argument("--num_domain", type=int, default=4, help="域数目")
parser.add_argument("--s_coef", type=float, default=0.02, help="情感影响因子")
args = parser.parse_args()


MDSD_PATH = os.getcwd()
print('MDSD_PATH', MDSD_PATH)
assert os.path.exists(MDSD_PATH)
DOMAINS = ('books', 'dvd', 'electronics', 'kitchen')#定义的域名，即数据集文件夹名称

def load_mdsd(domains, n_labeled=None):
    sorted_data_path = os.path.join(MDSD_PATH, 'sorted_data')  # 拼接sort_data路径
    print('loading data from {}'.format(sorted_data_path))
    texts = []#自定义列表，存储文本
    s_labels = []#存储情感名称
    d_labels = []#存储域名
    sentiments = ('positive', 'negative')#自定义元组，两类情感标签
    for d_id, d_name in enumerate(domains):#遍历数据集
        for s_id, s_name in zip((1, 0, -1), sentiments):
            fpath = os.path.join(sorted_data_path, d_name, s_name + '.review')#拼接文件路径：positive.review
            print(' - loading', d_name, s_name)
            count = 0#计数
            text = ''#自定义的空的文本变量
            in_review_text = False#自定义变量，用于定义当前是否处于文本读取区域
            with io.open(fpath, encoding='utf8', errors='ignore') as fr:#逐行读取文件
                for line in fr:
                    if '<review_text>' in line:#当某一行存在<review_text>时，将文本变量置为空，判定变量置为true，开始读取文本
                        text = ''#例如positive.review文件的38行
                        in_review_text = True
                        continue
                    if '</review_text>' in line:
                        in_review_text = False#停止读取文本
                        text = text.lower().replace('\n', ' ').strip()#转换成小写并将换行符转换为空格
                        text = re.sub(r'&[a-z]+;', '', text)#删除HTML实体
                        text = re.sub(r'\s+', ' ', text)#将多个连续空格替换为单个空格
                        texts.append(text)#将读取的文本加入列表
                        s_labels.append(s_id)#添加情感名称
                        d_labels.append(d_id)#添加域名
                        count += 1
                    if in_review_text:
                        text += line#读取文件时逐行添加进text变量中，还未添加进列表中
                    # labeled cutoff
                    if (s_id >= 0) and n_labeled and (count == n_labeled):#读取数量达到预设值时停止
                        break
            print(': %d texts' % count)
    print('data loaded')
    s_labels = np.asarray(s_labels, dtype='int')#列表转化成数组
    d_labels = np.asarray(d_labels, dtype='int')
    return texts, s_labels, d_labels

def _tvt_split(_seqs, _slabels, splits=(7, 2, 1)):#函数,添加d_label，s_label
    #train/val/test split for one single domain
    assert len(_seqs) == len(_slabels) #确保文本数与标签数目一致
    splits = np.asarray(splits)#转化为数组
    splits = np.cumsum([splits / float(splits.sum())])#计算累计和
    #print('splits:', splits, _seqs)
    # shuffle
    indices = [range(len(_seqs))]#创建一个包含从 0 到 _seqs 长度的整数序列的列表。
    np.random.shuffle(indices)#打乱索引
    _seqs = _seqs[indices]#使用洗牌后的索引重新排列 _seqs
    _slabels = _slabels[indices]#使用洗牌后的索引重新排列 _slabels
    #_dlabels = _dlabels[indices]
    # prepare data (balance data from all labels)
    X_train, ys_train, X_val, ys_val, X_test, ys_test = [],[],[],[],[],[]
    U = sorted(np.unique(_slabels))#获取 _slabels 中所有唯一标签，并将它们排序。0/1
    for slabel in sorted(np.unique(_slabels)):#遍历所有唯一标签
        seqs_ofs = _seqs[_slabels == slabel]#获取所有反例/正例
        slabels_ofs = _slabels[_slabels == slabel]
        seqs_ofs = seqs_ofs[0:args.sample_size]#截取序列
        slabels_ofs = slabels_ofs[0:args.sample_size]
        print('_seqs, _slabels', len(seqs_ofs), len(slabels_ofs))
        # split
        split_ats = np.asarray(splits * len(seqs_ofs), dtype=int)#按比例分配
        X_train.extend(seqs_ofs[:split_ats[0]])#7份
        X_val.extend(seqs_ofs[split_ats[0]:split_ats[1]])#2份
        X_test.extend(seqs_ofs[split_ats[1]:])#1份
        ys_train.extend(slabels_ofs[:split_ats[0]])#7份
        ys_val.extend(slabels_ofs[split_ats[0]:split_ats[1]])#2份
        ys_test.extend(slabels_ofs[split_ats[1]:])#1份
    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    X_test = np.asarray(X_test)
    ys_train = np.asarray(ys_train, dtype='int')
    ys_val = np.asarray(ys_val, dtype='int')
    ys_test = np.asarray(ys_test, dtype='int')

    print(' * X:', X_train.shape, X_val.shape, X_test.shape)
    print(' * ys:', ys_train.shape, ys_val.shape, ys_test.shape)
    return (X_train, X_val, X_test), (ys_train, ys_val, ys_test)
    #此时返回的训练集还是包含正反例的正序文件，还需再次打乱

def make_data():
    # domain & train/val/test split
    text, s_labels, d_labels = load_mdsd(domains=DOMAINS)
    # print('text,d_label,s_labels:', text[0:3], d_labels[0:3], s_labels[0:3])
    text = np.asarray(text)#列表元素长度不一致，无法转化为数组
    s_labels = np.asarray(s_labels, dtype=int)
    #print('s_label1-10', s_labels[0:10])
    d_labels = np.asarray(d_labels, dtype=int)
    print('labeled data: domain & train/val/test splitting')
    X_train, ys_train, yd_train = [], [], []
    X_val, ys_val, yd_val = [], [], []
    X_test_byd, ys_test_byd, yd_test_byd = {}, {}, {}
    ###################域分类
    for d_id, d_name in enumerate(DOMAINS):
        #print('条件：', (d_labels == d_id) & (s_labels != -1))
        texts_padded_ofd = text[(d_labels == d_id) & (s_labels != -1)]#获取属于该域且标签不为 -1 的序列和标签(文本内容）
        #print('texts_padded_ofd', texts_padded_ofd)
        slabels_ofd = s_labels[(d_labels == d_id) & (s_labels != -1)]
        #print(' * all:', texts_padded_ofd[0:10], slabels_ofd[0:10], d_labels[0:10])#每个域2000条，正反例子各有1000
        (X_train_ofd, X_val_ofd, X_test_ofd), (y_train_ofd, y_val_ofd, y_test_ofd) = _tvt_split(texts_padded_ofd, slabels_ofd)
        # train data (add this domain)按7：2：1划分
        print('  - X_train:', X_train_ofd.shape)
        X_train.extend(X_train_ofd)#将训练数据添加到 X_train 列表中
        #print("x_train[0]:", X_train[0],X_train[0].shape )
        ys_train.extend(y_train_ofd)
        yd_train.extend([d_id] * len(X_train_ofd))
        # val data
        X_val.extend(X_val_ofd)
        ys_val.extend(y_val_ofd)
        yd_val.extend([d_id] * len(X_val_ofd))
        # test data
        X_test_byd[d_id] = X_test_ofd
        ys_test_byd[d_id] = y_test_ofd
        yd_test_byd[d_id] = [d_id] * len(X_test_ofd)
    #X_train = np.asarray(X_train)  # 700*2*4=5600
    #print('  - X_train-shape:', X_train.shape)
    X_val = np.asarray(X_val)
    print('  - X_val-shape:', X_val.shape)
    ys_val= np.asarray(ys_val)  # 700*2*4=5600
    print('  - ys_val-shape:', ys_val.shape)
    yd_val = np.asarray(yd_val)
    print('  - yd_val-shape:', yd_val.shape)

    # combine test data from different domains
    X_test = np.concatenate([X_test_byd[idx] for idx in range(len(DOMAINS))])
    ys_test = np.concatenate([ys_test_byd[idx] for idx in range(len(DOMAINS))])
    yd_test = np.concatenate([yd_test_byd[idx] for idx in range(len(DOMAINS))])

    indices = list(range(len(X_train)))
    np.random.shuffle(indices)
    X_train = np.array(X_train)[indices]
    ys_train = np.array(ys_train)[indices]
    yd_train = np.array(yd_train)[indices]
    return X_train, X_val, X_test, ys_train, ys_val, ys_test, yd_train, yd_val, yd_test


class CustomTextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, s_labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.s_labels = s_labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        s_label = self.s_labels[index]

        # Tokenize text using the provided tokenizer
        inputs = self.tokenizer(text, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
        return inputs, label, s_label


class BertClassifier(nn.Module):
    def __init__(self, dropout=0.9):
        super(BertClassifier, self).__init__()
        #self.d_bert = BertModel.from_pretrained('bert-base-cased')
        #self.d_dropout = nn.Dropout(dropout)
        #self.d_linear = nn.Linear(768, 4)
        #self.d_relu = nn.ReLU()
        self.s_bert = BertModel.from_pretrained('bert-base-cased',
                                                output_hidden_states=True,
                                                output_attentions=True)
        self.s_dropout = nn.Dropout(dropout)
        self.s_linear = nn.Linear(768, 2)
        self.s_relu = nn.ReLU()

    def forward(self, input_id, mask):
        #_, d_pooled_output = self.d_bert(input_ids= input_id, attention_mask=mask,return_dict=False)
        #d_dropout_output = self.d_dropout(d_pooled_output)
        #d_linear_output = self.d_linear(d_dropout_output)
        #d_final_layer = self.d_relu(d_linear_output)

        _, s_pooled_output, s_all_hidden_states, _ = self.s_bert(input_ids=input_id, attention_mask=mask, return_dict=False)
        s_dropout_output = self.s_dropout(s_pooled_output)
        s_linear_output = self.s_linear(s_dropout_output)
        s_final_layer = self.s_relu(s_linear_output)
        return s_final_layer


def train(model, train_data, val_data, learning_rate, epochs):
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=args.batch_size)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    #criterion1 = nn.CrossEntropyLoss()
    criterion2 = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    if use_cuda:
        model = model.to(device)
        #criterion1 = criterion1.to(device)
        criterion2 = criterion2.to(device)
    for epoch_num in range(epochs):

        total_acc_train = 0
        total_loss_train = 0
        s_total_acc_train = 0
        s_total_loss_train = 0
      
        for train_input, train_label, s_train_label in tqdm(train_dataloader):
            train_label = train_label.to(device)
            s_train_label = s_train_label.to(device)
            mask = train_input['attention_mask'].to(device)
            input_id = train_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)
            #print('output', output)
            #print('train_label', train_label.long())
            #batch_loss = criterion1(output, train_label.long())
            
            #total_loss_train += batch_loss.item()
            batch_loss = criterion2(output, s_train_label.long())
            total_loss_train += batch_loss.item()

            #acc1 = (output.argmax(dim=1) == train_label).sum().item()
            acc2 = (output.argmax(dim=1) == s_train_label).sum().item()
            #total_acc_train += acc1
            total_acc_train += acc2

            #batch_loss = args.coef * batch_loss + args.s_coef * s_batch_loss
            model.zero_grad()
            batch_loss.backward()
            optimizer.step()

        total_acc_val = 0
        total_loss_val = 0
        #s_total_acc_val = 0
        #s_total_loss_val = 0
        with torch.no_grad():
            for val_input, val_label, s_val_label in val_dataloader:
                #val_label = val_label.to(device)
                s_val_label = s_val_label.to(device)
                mask = val_input['attention_mask'].to(device)
                input_id = val_input['input_ids'].squeeze(1).to(device)

                output = model(input_id, mask)

                #batch_loss = criterion1(output, val_label.long())
                batch_loss = criterion2(output, s_val_label.long())
                total_loss_val += batch_loss.item()
                #s_total_loss_val += s_batch_loss.item()

                #acc1 = (output.argmax(dim=1) == val_label).sum().item()
                acc2 = (output.argmax(dim=1) == s_val_label).sum().item()
                #total_acc_val += acc1
                total_acc_val += acc2
        print(
            f'Epochs: {epoch_num + 1} |Sentiment Train Loss: {total_loss_train / len(train_data): .3f} \
                | Sentiment Train Accuracy: {total_acc_train / len(train_data): .3f} \
                | Sentiment Val Loss: {total_loss_val / len(val_data): .3f} \
                | Sentiment Val Accuracy: {total_acc_val / len(val_data): .3f}')


def evaluate(model, test_data):
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.to(device)

    total_acc_test = 0
    #s_total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label, s_test_label in test_dataloader:
            #test_label = test_label.to(device)
            s_test_label = s_test_label.to(device)
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)

            output = model(input_id, mask)

            #acc = (output.argmax(dim=1) == test_label).sum().item()
            s_acc =  (output.argmax(dim=1) == s_test_label).sum().item()
            total_acc_test += acc
            #s_total_acc_test += s_acc
    print(f'Sentiment Test Accuracy: {total_acc_test / len(test_data): .3f}')
    

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# 设置随机数种子
setup_seed(1763)


# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
config = BertConfig.from_pretrained("bert-base-cased")
# Create an instance of CustomTextDataset
X_train, X_val, X_test, ys_train, ys_val, ys_test, yd_train, yd_val, yd_test = make_data()
train_custom_dataset = CustomTextDataset(X_train, yd_train.tolist(), ys_train.tolist(), tokenizer)
val_custom_dataset = CustomTextDataset(X_val, yd_val.tolist(), ys_val.tolist(), tokenizer)
test_custom_dataset = CustomTextDataset(X_test, yd_test.tolist(), ys_test.tolist(), tokenizer)
EPOCHS = 20
model = BertClassifier()
LR = 1e-6


for name, param in model.named_parameters():
    print(name, param.size())

"""
freeze_layers = ['bert.embeddings', 'bert.encoder']

for name, param in model.named_parameters():
    param.requires_grad = True
    for ele in freeze_layers:
        if ele in name:
            param.requires_grad = False
            break
"""
train(model, train_custom_dataset, val_custom_dataset, LR, EPOCHS)
evaluate(model, test_custom_dataset)


