import sys
import time
import torch
import argparse
import colab  # for colab env
import torch.nn.functional as F

from tqdm import tqdm
from datetime import datetime
from layers import create_masks
from torchmetrics import F1Score
from transformer import get_model
from scheduler import CosineWithWarmRestarts
from preprocess import create_data, create_files

# check gpu
if torch.cuda.is_available():
    cuda = True
    processor = "cuda"
else:
    cuda = False
    processor = "cpu"

# check env
modulename = 'colab'
if modulename in sys.modules:
    # gg colab env
    data_path = "/content/drive/MyDrive/Modelling/data/tmp.txt"
    log_train_path = "/content/drive/MyDrive/Modelling/weights/log_train.txt"
    model_path = "/content/drive/MyDrive/Modelling/weights/model.txt"
else:
    # local env
    data_path = "data/tmp.txt"
    log_train_path = "weights/log_train.txt"
    model_path = "weights/model"


def train_model(model, option, SOURCE, TARGET):
    print("Training model...")
    model.train()

    if option.checkpoint == True:
        cptime = time.time()

    # In case of removing pretrained model
    # if os.path.exists('weights/model'):
    #     os.remove('weights/model')

    # if os.path.exists('weights/log_train.txt'):
    #     os.remove('weights/log_train.txt')

    f = open(log_train_path, 'a')
    f.write(f'\nTrain at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
    f1_metric = F1Score(ignore_index=option.target_pad).to(option.cuda_device)

    for epoch in range(option.epochs):
        cptime = time.time()

        f.write(f"Epoch {epoch + 1}/{option.epochs}\n")
        print((f"Epoch {epoch + 1}/{option.epochs}:"))
        for iteration, batch in tqdm(enumerate(option.train), total=1500):
            source = batch.source.transpose(0, 1)
            target = batch.target.transpose(0, 1)
            target_input = target[:, :-1]
            source_mask, target_mask = create_masks(source, target_input, option)

            if option.cuda == True:
                source = source.to(option.cuda_device)
                target_input = target_input.to(option.cuda_device)
                source_mask = source_mask.to(option.cuda_device)
                target_mask = target_mask.to(option.cuda_device)

            preds = model(source, target_input, source_mask, target_mask)
            ys = target[:, 1:].contiguous().view(-1)
            if option.cuda == True:
                ys = ys.to(option.cuda_device)

            option.optimizer.zero_grad()

            loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys, ignore_index=option.target_pad)
            f1_score = f1_metric(preds.view(-1, preds.size(-1)), ys)
            loss.backward()
            option.optimizer.step()

            if option.scheduler == True:
                option.sched.step()

        f.write(f"Time: {time.time() - cptime}.\n");
        print(f"Time: {time.time() - cptime}.")
        f.write(f"Loss: {loss.item()}\n");
        print(f"Loss: {loss.item()}")
        f.write(f"F1 Score: {f1_score}\n");
        print(f"F1 Score: {f1_score}")

        if option.checkpoint == True:
            f.write(f"Save model after {epoch + 1} epoch(s).\n")
            print(f"Save model after {epoch + 1} epoch(s).")
            torch.save(model.state_dict(), model_path)

        option.train = create_data(option, SOURCE, TARGET, repeat=1)

        f.write("\n")
    f.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=bool, default=False)
    parser.add_argument('-data_file', type=str, default=data_path)
    parser.add_argument('-source_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-target_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-epochs', type=int, default=50)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=bool, default=cuda)
    parser.add_argument('-cuda_device', type=str, default=processor)
    parser.add_argument('-batch_size', type=int, default=100)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-threshold', type=int, default=0.94)
    parser.add_argument('-max_strlen', type=int, default=300)
    parser.add_argument('-checkpoint', type=bool, default=True)
    parser.add_argument('-scheduler', type=bool, default=True)

    option = parser.parse_args()

    SOURCE, TARGET = create_files(option)
    option.train = create_data(option, SOURCE, TARGET)

    print(f'Length of SOURCE vocab: {len(SOURCE.vocab)}, TARGET vocab: {len(TARGET.vocab)}')
    model = get_model(option, len(SOURCE.vocab), len(TARGET.vocab))

    option.optimizer = torch.optim.Adam(model.parameters(), lr=option.lr, betas=(0.9, 0.98), eps=1e-9)
    if option.scheduler == True:
        option.sched = CosineWithWarmRestarts(option.optimizer, T_max=option.train_len)

    if option.checkpoint == True:
        print("Model is saved at the end of each epoch.")

    train_model(model, option, SOURCE, TARGET)


if __name__ == "__main__":
    main()
