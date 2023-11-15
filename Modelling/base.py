import torch
from transformer import get_model
from preprocess import *
from beam_search import beam_search
from torch.autograd import Variable


# check gpu & time
if torch.cuda.is_available():
    cuda = True
    processor = "cuda"
else:
    cuda = False
    processor = "cpu"


class ArgumentOpt:
    def __init__(self):
        self.load_weights=True
        self.k=3
        self.source_lang="en_core_web_sm"
        self.target_lang="en_core_web_sm"
        self.d_model=512
        self.n_layers=6
        self.heads=8
        self.dropout=0.1
        self.max_strlen=300
        self.cuda=True
        self.cuda_device=processor

class SpellingCorrection:
    def __init__(self):
        self.option = ArgumentOpt()
        self.SOURCE, self.TARGET = create_files(self.option)
        self.model = get_model(self.option, len(self.SOURCE.vocab), len(self.TARGET.vocab))

    def translate_sentence(self, sentence):
        self.model.eval()
        indexed = []
        sentence = preprocess(sentence)
        for tok in sentence:
            indexed.append(self.SOURCE.vocab.stoi[tok])
        sentence = Variable(torch.LongTensor([indexed]))
        if self.option.cuda == True:
            sentence = sentence.to(self.option.cuda_device)
        sentence = beam_search(sentence, self.model, self.SOURCE, self.TARGET, self.option)
        return sentence.capitalize()

    def __call__(self, sentence):
        return self.translate_sentence(sentence)