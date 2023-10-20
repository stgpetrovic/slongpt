from encdec import Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
block_size = 8 # max length to send for training, context_length
batch_size = 32 # how many we can batch together for performance
max_iters = 3000
eval_interval = 300
lr = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32

torch.manual_seed(1337)

def ReadInput(f):
    with open(f, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def ValidateTokenizer(tknz):
  print(tknz)
  assert tknz.decode(tknz.encode("slon")) == "slon"

text = ReadInput('input/input.txt')
tknz = Tokenizer(text)
ValidateTokenizer(tknz)


def Split(data, ratio):
    n = int(ratio*len(data))
    return data[:n], data[n:]

def GetBatch(data, split):
    # batch contans data of inputs x and targets y
    ix = torch.randint(high=len(data[split]) - block_size, size=(batch_size,))
    x = torch.stack([data[split][i:i+block_size] for i in ix])
    y = torch.stack([data[split][i+1:1+i+block_size] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    """One head of egotism, aka, self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # BTC
        k = self.key(x)   # BTC
        w = q @ k.transpose(-2, -1) * C**-.5 # BTC @ BCT => BTT
        w = w.masked_fill(self.tril[:T,:T] == 0, float('-inf')) #BTT
        w = F.softmax(w, dim=-1) # BTT
        v = self.value(x) # BTC
        out = w @ v # BTT @ BTC => BTC
        return out


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(len(tknz), n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.sa_head = Head(n_embd)
        self.lm_head = nn.Linear(n_embd, len(tknz))

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) #BTC
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # T C
        x = tok_emb + pos_emb # BTC
        x = self.sa_head(x) #BTC
        logits = self.lm_head(x)  #B T vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is B,T array of indices in context
        for _ in range(max_new_tokens):
            logits, loss = self(idx[:, -block_size:])
            logits = logits[:, -1, :] # drop T, get BC
            # get probabilities
            probs = F.softmax(logits, dim=-1) # BC
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # B,1
            # append samplex index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # B,T+1
        return idx

def TryBigramLM(tknz, data, xb, yb):
    model = BigramLanguageModel()
    m = model.to(device)
    Train(tknz, model, data, 'train')

@torch.no_grad()
def EstimateLoss(model, data):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, )
        for k in range(eval_iters):
            x, y = GetBatch(data, split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def Train(tknz, model, data, split):
 optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
 for s in range(max_iters):
    if s % eval_interval == 0:
        losses = EstimateLoss(model, data)
        print(f"step {s}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample
    xb, yb = GetBatch(data, split)

    # evaluate loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

 print(loss.item())
 context = torch.zeros((1,1), dtype=torch.long, device=device)
 print(tknz.decode(model.generate(context, max_new_tokens=500)[0].tolist()))

def Main():

    data = torch.tensor(tknz.encode(text), dtype=torch.long)

    train_data, val_data = Split(data, 0.9)
    data = {'train': train_data, 'val': val_data}
    print('train size %d, validation size %d' % (len(train_data), len(val_data)))

    x = train_data[:block_size]
    y = train_data[:block_size+1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} target is {target}")

    xb, yb = GetBatch(data, 'train')
    print (f"shapes {x.shape} {y.shape}")
    print (xb)

    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f"when input is {context.tolist()} the target: {target}")
    TryBigramLM(tknz, data, xb, yb)





Main()
