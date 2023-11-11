import sys
# import colab

from torch import nn
from layers import *

# check env
modulename = 'colab'
if modulename in sys.modules:
    # gg colab env
    model_path = "/content/drive/MyDrive/Modelling/weights/model"
else:
    # local env
    model_path = "weights/model"


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, source, mask):
        x = self.embed(source)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)

    def forward(self, target, e_outputs, source_mask, target_mask):
        x = self.embed(target)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, source_mask, target_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab, d_model, N, heads, dropout)
        self.decoder = Decoder(trg_vocab, d_model, N, heads, dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self, source, target, source_mask, target_mask):
        e_outputs = self.encoder(source, source_mask)
        d_output = self.decoder(target, e_outputs, source_mask, target_mask)
        output = self.out(d_output)
        return output


def get_model(option, src_vocab, trg_vocab):
    assert option.d_model % option.heads == 0
    assert option.dropout < 1

    model = Transformer(src_vocab, trg_vocab, option.d_model, option.n_layers, option.heads, option.dropout)

    if option.load_weights is True:
        print("Loading pretrained model...")
        model.load_state_dict(torch.load(model_path, map_location=option.cuda_device))
        print("Finished.")
    else:
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    if option.cuda == True:
        model = model.to(option.cuda_device)
    return model