from encdec import Tokenizer
import torch
import torch.nn as nn
from torch.nn import functional as F
torch.manual_seed(1337)

def ReadInput(f):
    with open(f, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def ValidateTokenizer(tknz):
  print(tknz)
  assert tknz.decode(tknz.encode("slon")) == "slon"


def Split(data, ratio):
    n = int(ratio*len(data))
    return data[:n], data[n:]

def GetBatch(data, batch_size, block_size):
    # batch contans data of inputs x and targets y
    ix = torch.randint(high=len(data) - block_size, size=(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:1+i+block_size] for i in ix])
    return x, y

class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) #BTC
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
            logits, loss = self(idx)
            logits = logits[:, -1, :] # drop T, get BC
            # get probabilities
            probs = F.softmax(logits, dim=1) # BC
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1) # B,1
            # append samplex index to running sequence
            idx = torch.cat((idx, idx_next), dim=1) # B,T+1
        return idx

def TryBigramLM(tknz, data, batch_size, block_size, xb, yb):
    m = BigramLanguageModel(len(tknz))
    logits, loss = m(xb, yb)
    idx = torch.zeros((1,1), dtype=torch.long)
    print(tknz.decode(m.generate(idx, max_new_tokens=100)[0].tolist()))

    # more serious attempt
    Train(tknz, m, data, batch_size, block_size, 10000)

def Train(tknz, m, data, batch_size, block_size, steps):
 optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
 for s in range(steps):
    # sample
    xb, yb = GetBatch(data, batch_size, block_size)

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

 print(loss.item())
 idx = torch.zeros((1,1), dtype=torch.long)
 print(tknz.decode(m.generate(idx, max_new_tokens=500)[0].tolist()))

def Main():
    text = ReadInput('input/input.txt')
    tknz = Tokenizer(text)
    ValidateTokenizer(tknz)

    data = torch.tensor(tknz.encode(text), dtype=torch.long)

    train_data, val_data = Split(data, 0.9)
    print('train size %d, validation size %d' % (len(train_data), len(val_data)))

    block_size = 8 # max length to send for training, context_length
    x = train_data[:block_size]
    y = train_data[:block_size+1]
    for t in range(block_size):
        context = x[:t+1]
        target = y[t]
        print(f"when input is {context} target is {target}")

    batch_size = 32 # how many we can batch together for performance
    xb, yb = GetBatch(train_data, batch_size, block_size)
    print (f"shapes {x.shape} {y.shape}")
    print (xb)

    for b in range(batch_size):
        for t in range(block_size):
            context = xb[b, :t+1]
            target = yb[b, t]
            print(f"when input is {context.tolist()} the target: {target}")
    TryBigramLM(tknz, train_data, batch_size, block_size, xb, yb)





Main()
