import torch
from layers import no_peak_masks
import torch.nn.functional as F
import math


def init_vars(src, model, SRC, TRG, opt):
    init_tok = TRG.vocab.stoi['<sos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    e_output = model.encoder(src, src_mask)
    outputs = torch.LongTensor([[init_tok]])
    if opt.cuda == True:
        outputs = outputs.to(opt.cuda_device)
    trg_mask = no_peak_masks(1, opt)
    out = model.out(model.decoder(outputs, e_output, src_mask, trg_mask))
    out = F.softmax(out, dim=-1)
    probs, ix = out[:, -1].data.topk(opt.k)
    log_scores = torch.Tensor([math.log(prob) for prob in probs.data[0]]).unsqueeze(0)
    outputs = torch.zeros(opt.k, opt.max_strlen).long()
    if opt.cuda == True:
        outputs = outputs.to(opt.cuda_device)
    outputs[:, 0] = init_tok
    outputs[:, 1] = ix[0]
    e_outputs = torch.zeros(opt.k, e_output.size(-2),e_output.size(-1))
    if opt.cuda == True:
        e_outputs = e_outputs.to(opt.cuda_device)
    e_outputs[:, :] = e_output[0]
    return outputs, e_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, probs_ls, i, k, probability_threshold=.5):
    probs, ix = out[:, -1].data.topk(k)
    print(probs, 'probs')
    print(ix, 'ix')
    
    # thresholxding the generated result from model
    valid_indices = probs > probability_threshold
    valid_probs = probs[valid_indices]
    valid_ix = ix[valid_indices]
    
    flattened_probs = probs.view(-1)
    log_probs = torch.Tensor([math.log(p) for p in flattened_probs]).view(k, -1) + log_scores.transpose(0, 1)
    print(log_probs, 'log_probs')
    k_probs, k_ix = log_probs.view(-1).topk(k)
    print(k_probs, 'k_probs')
    print(k_ix, 'k_ix')

    row = k_ix // k
    print(row, 'row')
    col = k_ix % k
    print(col, 'col')

    print(outputs, 'outputs 0')
    outputs[:, :i] = outputs[row, :i]
    print(outputs, 'outputs 1')
    outputs[:, i] = ix[row, col]
    print(outputs, 'outputs 2')

    log_scores = k_probs.unsqueeze(0)
    print(log_scores, 'log_scores')
    return outputs, log_scores


def beam_search(src, model, SRC, TRG, opt):
    probs_ls = [[]]
    ind = None
    eos_tok = TRG.vocab.stoi['<eos>']
    src_mask = (src != SRC.vocab.stoi['<pad>']).unsqueeze(-2)
    outputs, e_outputs, log_scores = init_vars(src, model, SRC, TRG, opt)

    # for each state in frontier
    for i in range(2, opt.max_strlen):
        trg_mask = no_peak_masks(i, opt)
        out = model.out(model.decoder(outputs[:, :i], e_outputs, src_mask, trg_mask))
        out = F.softmax(out, dim=-1)

        # for each char in vocab
        outputs, log_scores = k_best_outputs(outputs, out, log_scores, probs_ls, i, opt.k)
        ones = (outputs == eos_tok).nonzero()  # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(opt.cuda_device)

        for vec in ones:
            i = vec[0]
            if sentence_lengths[i] == 0:  # First end symbol has not been found yet
                sentence_lengths[i] = vec[1]  # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == opt.k:
            alpha = 0.5
            div = 1/(sentence_lengths.type_as(log_scores) ** alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind.data[0]
            break
    s = ""

    if ind is None:
        length = (outputs[0] == eos_tok).nonzero()[0]
        for tok in outputs[0][1:length]:
            s += TRG.vocab.itos[tok]
        return s
    length = (outputs[ind] == eos_tok).nonzero()[0]
    for tok in outputs[ind][1:length]:
        s += TRG.vocab.itos[tok]
    return s