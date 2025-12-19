'''
source: https://medium.com/@saipragna.kancheti/nanogpt-a-small-scale-gpt-for-text-generation-in-pytorch-tensorflow-and-jax-641c4efefbd5
'''
import requests
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

class GPTBlock(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(GPTBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_size, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, 4 * embed_size),
            nn.GELU(),
            nn.Linear(4 * embed_size, embed_size)
        )
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x, mask):
        attn_out, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.ln1(x + attn_out)
        ff_out = self.feed_forward(x)
        return self.ln2(x + ff_out)

class NanoGPT(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_blocks):
        super(NanoGPT, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.blocks = nn.ModuleList([GPTBlock(embed_size, num_heads) for _ in range(num_blocks)])
        self.fc = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        mask = torch.triu(torch.ones(len(x), len(x)), diagonal=1).bool().to(x.device)
        for block in self.blocks:
            x = block(x, mask)
        return self.fc(x)

class TextDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx:idx+self.block_size]).to(device), 
            torch.tensor(self.data[idx+1:idx+self.block_size+1]).to(device)
            )

def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for inputs, targets in dataloader:
        #inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

class NanoGPTWithRegularization(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers, dropout_prob=0.5):
        super(NanoGPTWithRegularization, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

        # Add dropout and layer normalization
        self.dropout = nn.Dropout(dropout_prob)
        self.layer_norm = nn.LayerNorm(hidden_size)
    def forward(self, x):
        x = self.embedding(x)
        x = self.dropout(x)  # Apply dropout after embedding
        lstm_out, _ = self.lstm(x)
        lstm_out = self.layer_norm(lstm_out)  # Apply layer normalization
        x = self.fc(lstm_out)
        return x

def generate_text(model, seed_text, gen_length=100, temperature=1.0):
    model.eval()
    with torch.no_grad():
        # Encode seed text
        input_ids = torch.tensor([stoi[c] for c in seed_text], dtype=torch.long).unsqueeze(0).to(device)

        # Generate text
        for _ in range(gen_length):
            outputs = model(input_ids)
            logits = outputs[:, -1, :]
            probs = nn.functional.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        # Decode the generated text
        generated_text = ''.join([itos[int(idx)] for idx in input_ids[0]])
    return generated_text

if __name__ == '__main__':
    BLOCK_SIZE = 128
    BATCH_SIZE = 5

    url = "https://www.gutenberg.org/files/11/11-0.txt"
    response = requests.get(url)
    text = response.text
    #val_dataset = TextDataset(val_data, BLOCK_SIZE)

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = { ch:i for i,ch in enumerate(chars) }
    itos = { i:ch for i,ch in enumerate(chars) }
    train_data = [stoi[w] for w in text]
    train_dataset = TextDataset(train_data, BLOCK_SIZE)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = NanoGPT(vocab_size=vocab_size,
                    embed_size=20,
                    num_heads=2,
                    num_blocks=2
                    )
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    model.to(device=device)
    for epoch in range(10):
        loss = train(model, train_loader, optimizer, nn.CrossEntropyLoss())
        print(f'epoch = {epoch}, loss = {loss}')
    # Generate text with the trained model
    seed = "Alice"
    generated_sequence = generate_text(model, seed_text=seed, gen_length=1000)
    print(generated_sequence)