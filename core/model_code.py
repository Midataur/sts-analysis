import torch
import torch.nn as nn
from torch.nn import functional as F
from game_data import VOCABULARY, AUGMENTED_CARDS_LIST
from dataloading import SimpleDataset

class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        n_embed = config["n_embed"]
        n_heads = config["n_heads"]
        dropout = config["dropout"]

        head_size = n_embed // n_heads

        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

        # this is just a place to attach a hook
        self.attention_hook = nn.Identity()
        self.sanity_hook = nn.Identity()

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)

        wei = self.sanity_hook(wei)

        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        wei = self.attention_hook(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        n_heads = config["n_heads"]
        n_embed = config["n_embed"]
        dropout = config["dropout"]

        self.heads = nn.ModuleList([Head(config) for _ in range(n_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(
            self.proj(
                torch.cat([h(x) for h in self.heads], dim=-1)
            )
        )

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()

        n_embed, dropout = config["n_embed"], config["dropout"]

        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed), # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Commmunication followed by computation"""

    def __init__(self, config):
        super().__init__()
        n_embed = config["n_embed"]

        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)

        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        # residuals
        # don't use += as this breaks things
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# behold, my weird bespoke not-quite-a transformer
class SkipBot(nn.Module):
    def __init__(self, config, *args):
        super().__init__()
        n_embed = config["n_embed"]
        n_blocks = config["n_blocks"]
        self.n_cont = 6

        vocab_size = len(VOCABULARY)
        card_count = len(AUGMENTED_CARDS_LIST)

        self.state_token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.cont_embedding_transformation = nn.Linear(self.n_cont, n_embed, bias=True)
        self.choice_token_embedding_table = nn.Embedding(card_count, n_embed)

        # this is just a place to attach a hook
        self.embed_hook = nn.Identity()
        
        self.blocks = [Block(config) for _ in range(n_blocks)]
        self.blocks.append(nn.LayerNorm(n_embed))

        self.blocks = nn.Sequential(*self.blocks)

        # the output layer
        # projects the final vector down to the output dimension
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=True)
       
        # we shouldn't use this during training, only generation
        # this is because cross entropy loss already applies a softmax
        # and we don't want to apply that twice
        self.softmax = nn.Softmax(dim=1)

    def forward(self, cat, cont, choice, *args):
        # get embeddings
        tok_emb = self.state_token_embedding_table(cat) # (B, C, E)
        cont_emb = self.cont_embedding_transformation(cont) # (B, E)
        card_emb = self.choice_token_embedding_table(choice) # (B, C, E)

        #reshape cont_emb
        batchsize = tok_emb.shape[0]
        cont_emb = cont_emb.reshape((batchsize, 1, -1)) # (B, C, E)

        x = torch.concat((tok_emb, cont_emb, card_emb), dim=1)
        
        x = self.embed_hook(x)
        
        x = self.blocks(x) # apply a bunch of blocks (sa + feedforward) (B, C, E)

        # add all the vectors together
        # seems interesting
        x = torch.sum(x, axis=1) # (B, E)

        logits = self.lm_head(x) #(B, output_dim)

        return logits
    
    def gen_from_states(self, states, choices, config, return_logits=False):
        # process the states
        dataset = SimpleDataset(states, choices, config, verbose=False)

        # get the prediction
        cat_tensor = dataset.state_cat
        cont_tensor = dataset.state_cont
        choice_tensor = dataset.card_choices

        logits = self(cat_tensor, cont_tensor, choice_tensor) # (B, output_dim)

        if return_logits:
            return logits

        return self.softmax(logits)
    
    def get_loss(self):
        return nn.CrossEntropyLoss()

# behold, my weird bespoke not-quite-a transformer (2nd edition)
class V2(nn.Module):
    def __init__(self, config, *args):
        super().__init__()
        n_embed = config["n_embed"]
        n_blocks = config["n_blocks"]
        self.n_cont = 12

        vocab_size = len(VOCABULARY)

        self.state_token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.cont_embedding_transformation = nn.Linear(self.n_cont, n_embed, bias=True)

        # this is just a place to attach a hook
        self.embed_hook = nn.Identity()
        
        self.blocks = [Block(config) for _ in range(n_blocks)]
        self.blocks.append(nn.LayerNorm(n_embed))

        self.blocks = nn.Sequential(*self.blocks)

        # the output layer
        # projects the final vector down to the output dimension
        self.lm_head = nn.Linear(n_embed, 1, bias=True)
       
        # we shouldn't use this during training, only generation
        # this is because cross entropy loss already applies a softmax
        # and we don't want to apply that twice
        self.sigmoid = nn.Sigmoid()

    def forward(self, cat, cont, *args):
        # get embeddings
        tok_emb = self.state_token_embedding_table(cat) # (B, C, E)
        cont_emb = self.cont_embedding_transformation(cont) # (B, E)

        #reshape cont_emb
        batchsize = tok_emb.shape[0]
        cont_emb = cont_emb.reshape((batchsize, 1, -1)) # (B, C, E)

        x = torch.concat((tok_emb, cont_emb), dim=1)
        
        x = self.embed_hook(x)
        
        x = self.blocks(x) # apply a bunch of blocks (sa + feedforward) (B, C, E)

        # add all the vectors together
        # seems interesting
        x = torch.sum(x, axis=1) # (B, E)

        logit = self.lm_head(x).reshape((-1)) # B

        return logit
    
    def gen_from_states(self, states, choices, config, return_logits=False):
        raise Exception("Not implemented")
    
    def get_loss(self):
        return nn.BCEWithLogitsLoss()

MODELS = {
    "skip-bot": SkipBot,
    "v2": V2
}