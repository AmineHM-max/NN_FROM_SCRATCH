import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameters
batch_size=32
block_size=8
max_iters=3000
eval_interval = 300
learning_rate=1e-2
device= 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters=200

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
  return x,y


@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



#Biagram model

class BiagramLanguageModel(nn.Module) :
  def __init__(self,vocab_size) :
      super().__init__()
      self.token_embedding_table=nn.Embedding(vocab_size,vocab_size)

  def forward(self,idx,targets=None) :
    logits=self.token_embedding_table(idx)
    if targets is not None :
        B,T,C=logits.shape
        logits=logits.view(B*T,C)
        targets=targets.view(B*T)
        loss=F.cross_entropy(logits,targets)
    else :
      loss=None
    return logits,loss
  def generate(self,idx,max_tokens) :
    # idx is the context token of shape (B,T)
    for _ in range(max_tokens) :
      logits,loss=self(idx) #(B,T,C)
      logits=logits[:,-1,:] #(B,C)
      probs=F.softmax(logits,dim=-1) #(B,C)
      idx_next=torch.multinomial(probs,num_samples=1) #(B,1)
      idx=torch.cat([idx,idx_next],dim=1)
    return idx


model=BiagramLanguageModel(vocab_size)
m=model.to(device)

optimizer=torch.optim.AdamW(m.parameters(),lr=learning_rate)



for i in range(max_iters) :
# every once in a while evaluate the loss on train and val sets
    if i % eval_interval == 0:
        losses = estimate_loss(model)
        print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_tokens=500)[0].tolist()))     