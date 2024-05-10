import torch.nn as nn
import torch
import math
from typing import List, Optional, Tuple
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


MAX_CHARS = 25
OUTPUT_MAX_LEN = MAX_CHARS #+ 2  # <GO>+groundtruth+<END>
charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
# cdict = {c:i for i,c in enumerate(c_classes)}
# icdict = {i:c for i,c in enumerate(c_classes)}


def font2int(font:str) -> int:
    if 'times' in font:
        return 0
    else:
        return 1

class Tokenizer:
    
    def __init__(self, charset: str):
        self.BOS = '[B]'
        self.EOS = '[E]'
        self.PAD = '[P]'
        self.specials_first = (self.EOS,)
        self.specials_last = (self.BOS, self.PAD)
        self._itos = self.specials_first + tuple(charset) + self.specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in self.specials_first + self.specials_last]

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: str) -> List[int]:
        return [self._stoi[s] for s in tokens]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> str:
        tokens = [self._itos[i] for i in token_ids]
        return ''.join(tokens) if join else tokens
    
    def encode(self, labels: List[str], device: Optional[torch.device] = None) -> Tensor:
        
        batch = [torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
                 for y in labels]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)
    



class Word_Attention(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Word_Attention, self).__init__()
        self.linear_query = nn.Linear(input_size, hidden_size)
        self.linear_key = nn.Linear(input_size, hidden_size)
        self.linear_value = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        query = self.linear_query(x)
        key = self.linear_key(x)
        value = self.linear_value(x)
        
        # Calculate attention scores
        scores = query @ key.transpose(-2, -1)
        scores = self.softmax(scores)
        
        # Calculate weighted sum of the values
        word_embedding = scores @ value
        return word_embedding



class CharacterEncoder(nn.Module):
    
    """
    args:
        input_size: charset length
        hidden_size: 512
        max_seq_len: 25
    """
    
    def __init__(self, input_size, hidden_size, max_seq_len):
        super(CharacterEncoder, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.attention = Word_Attention(hidden_size, hidden_size)

        self.embedding_dim = hidden_size
        self.max_seq_len = max_seq_len
        self.positional_encoding = self.get_positional_encoding()

    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)
        #Remove positional encoding for ablation study
        x += self.positional_encoding[:x.size(1), :].to(x.device)
        word_embedding = self.attention(x)
        
        return torch.sum(word_embedding, dim=1)
    
    def get_positional_encoding(self):
        positional_encoding = torch.zeros(self.max_seq_len, self.embedding_dim)
        for pos in range(self.max_seq_len):
            for i in range(0, self.embedding_dim, 2):
                positional_encoding[pos, i] = math.sin(pos / (10000 ** (i / self.embedding_dim)))
                positional_encoding[pos, i + 1] = math.cos(pos / (10000 ** ((i + 1) / self.embedding_dim)))
        return positional_encoding
    
    
    
