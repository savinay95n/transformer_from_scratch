import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, lang_src, lang_tgt, seq_len) -> None:
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.lang_src = lang_src
        self.lang_tgt = lang_tgt
        
        self.sos_token = torch.tensor([tokenizer_src.token_to_id(["[SOS]"])], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(["[EOS]"])], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(["[PAD]"])], dtype=torch.int64)
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.lang_src]
        tgt_text = src_target_pair['translation'][self.lang_tgt]
        
        # Tokenizer will first split the text into tokens and then convert those tokens to their corresponding ids in the vocabulary. 
        #The result is a list of integers representing the tokenized text.
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        # pad the input and target sequences to the same length (seq length)
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 # -2 for sos and eos tokens
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 # -1 for eos token
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError(f"Sequence length {self.seq_len} is too small for the given input and target texts.")  
        
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
    
        assert encoder_input.size(0) == self.seq_len, f"Encoder input sequence length {encoder_input.size(0)} does not match the specified sequence length {self.seq_len}."
        assert decoder_input.size(0) == self.seq_len, f"Decoder input sequence length {decoder_input.size(0)} does not match the specified sequence length {self.seq_len}."
        assert label.size(0) == self.seq_len, f"Label sequence length {label.size(0)} does not match the specified sequence length {self.seq_len}."
        
        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(), # (1, 1, seq_len)
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.size(0)), # (1, 1, seq_len) & (1, seq_len, seq_len)
            "label": label,
            "src_text": src_text,
            "tgt_text": tgt_text
        }
        
def causal_mask(size):
    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0