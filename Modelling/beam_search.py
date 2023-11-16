import torch
from layers import no_peak_masks
import torch.nn.functional as F
import math


def init_vars(source, model, SOURCE, TARGET, option):
    # compute encoder output
    source_mask = (source != SOURCE.vocab.stoi['<pad>']).unsqueeze(-2)
    encoder_output = model.encoder(source, source_mask)  # (batch_size, seq_len, d_model)
    encoder_outputs = torch.zeros(option.k, encoder_output.size(-2), encoder_output.size(-1))

    if option.cuda: encoder_outputs = encoder_outputs.to(option.cuda_device)

    encoder_outputs[:, :] = encoder_output[0]

    # compute decoder & model output
    init_token = TARGET.vocab.stoi['<sos>']
    outputs = torch.LongTensor([[init_token]])  # [["encoded_eos"]]

    if option.cuda == True:
        outputs = outputs.to(option.cuda_device)

    # Transformer's output probabilities
    target_mask = no_peak_masks(1, option)
    model_out = F.softmax(model.out(model.decoder(outputs, encoder_output, source_mask, target_mask)), dim=-1)  # decode and ff target
    probalities, indices = model_out[:, -1].topk(option.k)
    log_scores = torch.log(probalities).unsqueeze(0)

    outputs = torch.zeros(option.k, option.max_strlen).long()
    if option.cuda == True:
        outputs = outputs.to(option.cuda_device)
    outputs[:, 0] = init_token
    outputs[:, 1] = indices[0]
    return outputs, encoder_outputs, log_scores


def k_best_outputs(outputs, out, log_scores, i, k):
    probalities, indices = out[:, -1].data.topk(k)
    log_probalities = torch.Tensor([math.log(p) for p in probalities.data.view(-1)]).view(k, -1) + log_scores.transpose(0,1)
    k_probalities, k_indices = log_probalities.view(-1).topk(k)
    row = k_indices // k
    col = k_indices % k
    outputs[:, :i] = outputs[row, :i]
    outputs[:, i] = indices[row, col]
    log_scores = k_probalities.unsqueeze(0)
    return outputs, log_scores


def beam_search(source, model, SOURCE, TARGET, option):
    outputs, encoder_outputs, log_scores = init_vars(source, model, SOURCE, TARGET, option)

    eos_token = TARGET.vocab.stoi['<eos>']
    source_mask = (source != SOURCE.vocab.stoi['<pad>']).unsqueeze(-2)

    ind = None
    for i in range(2, option.max_strlen):
        target_mask = no_peak_masks(i, option)
        model_out = F.softmax(model.out(model.decoder(outputs[:,:i], encoder_outputs, source_mask, target_mask)), dim=-1)
        outputs, log_scores = k_best_outputs(outputs, model_out, log_scores, i, option.k)

        ones = (outputs == eos_token).nonzero() # Occurrences of end symbols for all input sentences.
        sentence_lengths = torch.zeros(len(outputs), dtype=torch.long).to(option.cuda_device)

        for vec in ones:
            i = vec[0]
            if sentence_lengths[i]==0: # First end symbol has not been found yet
                sentence_lengths[i] = vec[1] # Position of first end symbol

        num_finished_sentences = len([s for s in sentence_lengths if s > 0])

        if num_finished_sentences == option.k:
            alpha = 0.7
            div = 1/(sentence_lengths.type_as(log_scores)**alpha)
            _, ind = torch.max(log_scores * div, 1)
            ind = ind[0]
            break

    s = ""
    if ind is None:
        length = (outputs[0]==eos_token).nonzero()[0]
        for token in outputs[0][1:length]:
            s += TARGET.vocab.itos[token]
        return s

    length = (outputs[ind]==eos_token).nonzero()[0]
    for token in outputs[ind][1:length]:
        s += TARGET.vocab.itos[token]
    return s