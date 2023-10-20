class Tokenizer:
    atoi = None
    itoa = None

    def __init__(self, text):
        unique = sorted(list(set(text)))
        self.atoi = {ch:i for i, ch in enumerate(unique)}
        self.itoa = {i:ch for i, ch in enumerate(unique)}

    def encode(self, s):
        return [self.atoi[a] for a in s]

    def decode(self, n):
        return ''.join([self.itoa[i] for i in n])

    def __len__(self):
        return len(self.atoi)

    def __repr__(self):
        return 'vocab(%d): %s' % (len(self.atoi), self.atoi.keys())

