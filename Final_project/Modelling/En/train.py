import os
import argparse
import time
import torch
from transformer import get_model
from scheduler import CosineWithWarmRestarts
from preprocess import *
from layers import *
import torch.nn.functional as F
from torchmetrics import F1Score


def train_model(model, option, SOURCE, TARGET):
    print("Training model...")
    model.train()

    if option.checkpoint == True:
        cptime = time.time()

    if os.path.exists('weights/model'):
        os.remove('weights/model')

    if os.path.exists('weights/log_train.txt'):
        os.remove('weights/log_train.txt')

    f = open('weights/log_train.txt', 'w')
    f1_metric = F1Score(ignore_index=option.trg_pad).to(option.cuda_device)

    for epoch in range(option.epochs):
        cptime = time.time()

        f.write(f"Epoch {epoch + 1}/{option.epochs}\n")
        print((f"Epoch {epoch + 1}/{option.epochs}:"))

        for _, batch in enumerate(option.train): 
            src = batch.src.transpose(0,1)
            trg = batch.trg.transpose(0,1)
            trg_input = trg[:, :-1]
            src_mask, trg_mask = create_masks(src, trg_input, option)

            if option.cuda == True:
                src = src.to(option.cuda_device)
                trg_input = trg_input.to(option.cuda_device)
                src_mask = src_mask.to(option.cuda_device)
                trg_mask = trg_mask.to(option.cuda_device)

            preds = model(src, trg_input, src_mask, trg_mask)
            print(trg)
            print(preds)
            ys = trg[:, 1:].contiguous().view(-1)
            if option.cuda == True:
                ys = ys.to(option.cuda_device)

            option.optionimizer.zero_grad()
            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=option.trg_pad)
            f1_score = f1_metric(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            option.optionimizer.step()

            if option.scheduler == True: 
                option.sched.step()

        f.write(f"Time: {time.time() - cptime}.\n"); print(f"Time: {time.time() - cptime}.")
        f.write(f"Loss: {loss.item()}\n"); print(f"Loss: {loss.item()}")
        f.write(f"F1 Score: {f1_score}\n"); print(f"F1 Score: {f1_score}")

        if option.checkpoint == True:
            f.write(f"Save model after {epoch + 1} epoch(s).\n")
            print(f"Save model after {epoch + 1} epoch(s).")
            torch.save(model.state_dict(), 'weights/model')
        
        option.train = create_data(option, SOURCE, TARGET, repeat=1)

        f.write("\n")
        print()
    
    f.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=bool, default=False)
    parser.add_argument('-data_file', type=str, default="data/train.txt")
    parser.add_argument('-source_language', type=str, default="en_core_web_sm")
    parser.add_argument('-target_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-epochs', type=int, default=100)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-cuda_device', type=str, default="cpu")
    parser.add_argument('-batch_size', type=int, default=200)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-threshold', type=int, default=0.94)
    parser.add_argument('-max_strlen', type=int, default=300)
    parser.add_argument('-checkpoint', type=bool, default=True)
    parser.add_argument('-scheduler', type=bool, default=True)

    option = parser.parse_args()

    SOURCE, TARGET = create_files(option)
    print('Create file complete')
    start = time.time()
    option.train = create_data(option, SOURCE, TARGET)
    print(f'Create data complete in {start-time.time()}')

    print(f'Length of SOURCE vocab: {len(SOURCE.vocab)}, TARGET vocab: {len(TARGET.vocab)}')
    
    model = get_model(option, len(SOURCE.vocab), len(TARGET.vocab))
    print(model)

    option.optionimizer = torch.optionim.Adam(model.parameters(), lr=option.lr, betas=(0.9, 0.98), eps=1e-9)
    if option.scheduler == True:
        option.sched = CosineWithWarmRestarts(option.optionimizer, T_max=option.train_len)

    if option.checkpoint == True:
        print("Model is saved at the end of each epoch.")

    train_model(model, option, SOURCE, TARGET)


if __name__ == "__main__":
    main()
