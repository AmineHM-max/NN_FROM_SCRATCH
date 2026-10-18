import torch
import torch.nn as nn
import torch.nn.functional as F
import time

#hyperparameters
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 5000
learning_rate = 3e-4
device ='cuda'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
#----------------

torch.manual_seed(1337)

with open('input.txt','r') as f :
    text=f.read()


# building simple tokenizer where we gonna map each integer to a character in our data set
chars=sorted(list(set(text)))
vocab_size=len(chars)
stoi={s : i for i,s in enumerate(chars)} # mapping char in chars  -> to int
itos={i : s for s,i in stoi.items()} # mapping int to -> char in chars
encode = lambda text : [stoi[w] for w in text] # encoder take tex : str -> list of integers
decode = lambda l : ''.join([itos[i] for i in l]) # decoder take list of integers -> text : str



# train and test splits
data=torch.tensor(encode(text)) # creating dataset

n=int(0.9*len(data)) # 90 % train and 10 % validation
train_data=data[:n]
val_data=data[n:]


def get_batch(split : str)  :
  data = train_data if split == 'train' else val_data
  ix=torch.randint(len(data)-block_size,(batch_size,))
  x=torch.stack([data[i:i+block_size] for i in ix])
  y=torch.stack([data[i+1:i+1+block_size] for i in ix])

  x = x.to(device)
  y = y.to(device)
  return x,y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module) :
   """a class for creating a single Attention Head"""
   def __init__(self,head_size):
      super().__init__()
      self.query=nn.Linear(n_embd,head_size,bias=False)
      self.key=nn.Linear(n_embd,head_size,bias=False)
      self.value=nn.Linear(n_embd,head_size,bias=False)
      self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))

   def forward(self,x) :
      B,T,C=x.shape
      Q=self.query(x) # x @ W_q -> Q of shape (B,T,head_size)
      K=self.key(x) # x @ W_k -> K of shape (B,T,head_size)
      V=self.value(x) # x @ W_v -> V of shape (B,T,head_size)
      wei =  Q @ K.mT / K.shape[-1] ** 0.5
      wei = wei.masked_fill(self.tril[:T, :T]==0,float('-inf'))
      wei=F.softmax(wei,dim=-1) # (B,T,T)
      out= wei @ V #(B,T,head_size)
      return out


class MultiHeadAttention(nn.Module) :
     """ multiple head of attention in parallel """
     def __init__(self,num_heads,head_size):
        super().__init__() 
        self.heads=nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
     def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out 

class FeedFoward(nn.Module) :
   def __init__(self,n_embd) :
      super().__init__()
      self.net=nn.Sequential(
         nn.Linear(n_embd, n_embd * 4),
         nn.ReLU(),
         nn.Linear(n_embd*4,n_embd),
         nn.Dropout(dropout)
      )
   def forward(self,x) :
        return self.net(x)


class Block(nn.Module) :
   """ transfomer block """

   def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

   def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x



      






#GPT

class GPT(nn.Module) :
  def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd,n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)


  def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

  def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

start = time.perf_counter()

model=GPT()
m=model.to(device)

print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(m)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

end = time.perf_counter()



print(f"Execution time: {end - start:.4f} seconds")  