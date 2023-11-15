import copy
import torch
from torch import nn
from layers import Embedder, PositionalEncoder, EncoderLayer, DecoderLayer, Norm


# check env
colab = False
if colab:
    # gg colab env
    model_path = "/content/drive/MyDrive/Modelling/weights/model"
else:
    # local env
    model_path = "weights/model"


def get_clones(module, n_layers):
    return nn.ModuleList([copy.deepcopy(module) for i in range(n_layers)])


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)

    def forward(self, source, mask):
        x = self.embed(source)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, heads, dropout):
        super().__init__()
        self.n_layers = n_layers
        self.embed = Embedder(vocab_size, d_model)
        self.pe = PositionalEncoder(d_model)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), n_layers)
        self.norm = Norm(d_model)

    def forward(self, target, encoder_outputs, source_mask, target_mask):
        x = self.embed(target)
        x = self.pe(x)
        for i in range(self.n_layers):
            x = self.layers[i](x, encoder_outputs, source_mask, target_mask)
        return self.norm(x)



########################################################################################################################
class Transformer(nn.Module):
    def __init__(self, source_vocab, target_vocab, d_model, n_layers, heads, dropout):
        super().__init__()
        self.encoder = Encoder(source_vocab, d_model, n_layers, heads, dropout)
        self.decoder = Decoder(target_vocab, d_model, n_layers, heads, dropout)
        self.out = nn.Linear(in_features=d_model, out_features=target_vocab)

    def forward(self, source, target, source_mask, target_mask):
        encoder_outputs = self.encoder(source, source_mask)
        d_output = self.decoder(target, encoder_outputs, source_mask, target_mask)
        output = self.out(d_output)
        return output


def get_model(option, source_vocab, target_vocab):
    assert option.d_model % option.heads == 0
    assert option.dropout < 1

    model = Transformer(source_vocab, target_vocab, option.d_model, option.n_layers, option.heads, option.dropout)

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