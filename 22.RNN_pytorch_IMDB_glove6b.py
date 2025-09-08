import os
import re
import time
from itertools import chain

import gensim
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

TAG_RE = re.compile(r'<[^>]+>')


def preprocess_text(sen):
    # Removing html tags
    sentence = TAG_RE.sub('', sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


def readIMDB(path, seg):
    classes = ['pos', 'neg']
    data = []
    for label in classes:
        files = os.listdir(os.path.join(path, seg, label))
        for file in files:
            with open(os.path.join(path, seg, label, file), 'r', encoding='utf8') as rf:
                review = rf.read().replace('\\n', '')
                if label == 'pos':
                    data.append([preprocess_text(review), 1])
                elif label == 'neg':
                    data.append([preprocess_text(review), 0])
    return data


def tokenizer(text):
    return [tok.lower() for tok in text.split(' ')]  # 簡單使用空格來斷詞


train_data = readIMDB('/Users/leonjye/Documents/aclImdb', 'train')
test_data = readIMDB('/Users/leonjye/Documents/aclImdb', 'test')

train_tokenized = []
test_tokenized = []
for review, score in train_data:
    train_tokenized.append(tokenizer(review))
for review, score in test_data:
    test_tokenized.append(tokenizer(review))

vocab = set(chain(*train_tokenized))  # 把tokenized 所有字給串起來
vocab_size = len(vocab)

### define mapping between word and index ###
word_to_idx = {word: i+1 for i, word in enumerate(vocab)}
word_to_idx['<unk>'] = 0
idx_to_word = {i+1: word for i, word in enumerate(vocab)}
idx_to_word[0] = '<unk>'


def encode_samples(tokenized_samples):  # use word index mapping to encode token
    features = []
    for sample in tokenized_samples:
        feature = []
        for token in sample:
            if token in word_to_idx:
                feature.append(word_to_idx[token])
            else:
                feature.append(0)
        features.append(feature)
    return features


def pad_samples(features, maxlen=350, PAD=0):  # 截長補短 讓長度一致，這裡固定文章長度為maxlen=350
    padded_features = []
    for feature in features:
        if len(feature) >= maxlen:
            padded_feature = feature[:maxlen]
        else:
            padded_feature = feature
            while (len(padded_feature) < maxlen):
                padded_feature.append(PAD)
        padded_features.append(padded_feature)
    return padded_features


### 將token轉成index 並轉成 pytorch tensor ###
train_features = torch.tensor(pad_samples(encode_samples(train_tokenized)))
train_labels = torch.tensor([score for _, score in train_data])
test_features = torch.tensor(pad_samples(encode_samples(test_tokenized)))
test_labels = torch.tensor([score for _, score in test_data])

### split validation set from train_feature ###
# 從train中抽出500筆nep&500筆pos的資料當val
val_features = torch.cat((train_features[:500], train_features[-500:]), 0)
val_labels = torch.cat((train_labels[:500], train_labels[-500:]), 0)
train_features = train_features[500:-500]
train_labels = train_labels[500:-500]

### create pytorch dataloader ###
batch_size = 36
train_set = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True)
val_set = torch.utils.data.TensorDataset(val_features, val_labels)
val_iter = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False)

test_set = torch.utils.data.TensorDataset(test_features, test_labels)
test_iter = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False)

### load word2vec model ###
# pre-train model download from: https://github.com/stanfordnlp/GloVe
# preprocess:https://stackoverflow.com/questions/51323344/cant-load-glove-6b-300d-txt
wvmodel = gensim.models.KeyedVectors.load_word2vec_format(
    'glove.6B.100d.w2vformat.txt', binary=False, encoding='utf-8')

# map golve pretrain weight to pytorch embedding pretrain weight
embed_size = 100
# given 0 if the word is not in glove
weight = torch.zeros(vocab_size+1, embed_size)
for i in range(len(wvmodel.index_to_key)):
    try:
        # transfer to our word2ind
        index = word_to_idx[wvmodel.index_to_key[i]]
    except:
        continue
    weight[index, :] = torch.from_numpy(
        wvmodel.get_vector(wvmodel.index_to_key[i]))

### build model ###


class RNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 bidirectional, weight, labels, **kwargs):
        super(RNN, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding.from_pretrained(weight)
        self.embedding.weight.requires_grad = False
        # self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=self.num_hiddens,
                            num_layers=num_layers, bidirectional=self.bidirectional,
                            batch_first=True,
                            dropout=0.3)

        if self.bidirectional:
            self.linear1 = nn.Linear(num_hiddens * 4, labels)
        else:
            self.linear1 = nn.Linear(num_hiddens * 2, labels)

    def forward(self, inputs):
        embeddings = self.embedding(inputs)
        # states, hidden = self.lstm(embeddings.permute([1, 0, 2]))
        states, hidden = self.lstm(embeddings)
        # if it's bidirectional, choose first and last output
        encoding = torch.cat([states[0], states[-1]], dim=1)
        outputs = self.linear1(encoding)

        return outputs


num_epochs = 10
num_hiddens = 100
num_layers = 2
bidirectional = True
labels = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = RNN(vocab_size=(vocab_size+1), embed_size=embed_size,
          num_hiddens=num_hiddens, num_layers=num_layers,
          bidirectional=bidirectional, weight=weight,
          labels=labels)

print(net)

net.to(device)
loss_function = nn.CrossEntropyLoss()  # ~ nn.LogSoftmax()+nn.NLLLoss()
optimizer = optim.Adam(net.parameters())


def train(net, num_epochs, loss_function, optimizer, train_iter, val_iter):
    for epoch in range(num_epochs):
        start = time.time()
        train_loss, val_losses = 0, 0
        train_acc, val_acc = 0, 0
        n, m = 0, 0
        net.train()
        for feature, label in train_iter:
            n += 1
            optimizer.zero_grad()
            feature = feature.to(device)
            label = label.to(device)

            score = net(feature)
            loss = loss_function(score, label)
            loss.backward()
            optimizer.step()
            train_acc += accuracy_score(torch.argmax(
                score.cpu().data, dim=1), label.cpu())
            train_loss += loss

        with torch.no_grad():
            net.eval()
            for val_feature, val_label in val_iter:
                m += 1
                val_feature = val_feature.to(device)
                val_label = val_label.to(device)
                val_score = net(val_feature)
                val_loss = loss_function(val_score, val_label)
                val_acc += accuracy_score(torch.argmax(
                    val_score.cpu().data, dim=1), val_label.cpu())
                val_losses += val_loss

        runtime = time.time() - start
        print('epoch: %d, train loss: %.4f, train acc: %.2f, val loss: %.4f, val acc: %.2f, time: %.2f' %
              (epoch, train_loss.data/n, train_acc/n, val_losses.data/m, val_acc/m, runtime))

    # save final model
    state = {
        'epoch': epoch,
        'state_dict': net.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    # torch.save(state, os.path.join(model_save_path,'last_model.pt'))


def predict(net, test_iter):
    # state = torch.load(os.path.join(cwd,'checkpoint','epoch10_maxlen300_embed200.pt'),map_location=torch.device('cpu'))
    # net.load_state_dict(state['state_dict'])
    pred_list = []
    true_list = []
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        net.eval()
        for batch, label in test_iter:
            output = net(batch.to(device))
            pred_list.extend(torch.argmax(
                softmax(output), dim=1).cpu().numpy())
            true_list.extend(label.cpu().numpy())

    acc = accuracy_score(pred_list, true_list)
    print('test acc: %f' % acc)

    return acc, pred_list, true_list


print('start to train...')
train(net, num_epochs, loss_function, optimizer, train_iter, val_iter)

print('start to predict test set...')
acc, pred_list, true_list = predict(net, test_iter)
