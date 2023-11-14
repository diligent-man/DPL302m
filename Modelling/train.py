import sys
import time
# from sched import scheduler

import pytz
import torch
import argparse
import torch.nn.functional as F

from tqdm import tqdm
from pprint import pprint as pp
from datetime import datetime
from layers import create_masks
from torchmetrics import F1Score
from transformer import get_model
from scheduler import CosineWithWarmRestarts
from preprocess import create_data, create_files
from torch.utils.tensorboard import SummaryWriter

# print(f"Num of threads: {torch.get_num_threads()}")
# print(f"Num of inter-operations in each thread: {torch.get_num_interop_threads()}")
torch.set_num_threads(4)
torch.set_num_interop_threads(2)


# check gpu & time
if torch.cuda.is_available():
    cuda = True
    processor = "cuda"
else:
    cuda = False
    processor = "cpu"

# check env
colab = False
if colab:
    # gg colab env
    data_path = "/content/drive/MyDrive/Modelling/data/tmp.txt"
    log_train_path = "/content/drive/MyDrive/Modelling/weights/log_train.txt"
    model_path = "/content/drive/MyDrive/Modelling/weights/model"

    # get time at training
    local_time = datetime.now()
    # Define the local timezone
    local_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
    # Define the GMT+7 timezone
    gmt_plus_7 = pytz.timezone('Etc/GMT-12')
    # Localize the current time
    local_time = local_timezone.localize(local_time)
    # Convert to GMT+7
    registered_time = local_time.astimezone(gmt_plus_7).strftime("%d/%m/%Y %H:%M:%S")
else:
    # local env
    data_path = "data/tmp.txt"
    log_train_path = "weights/log_train.txt"
    model_path = "weights/model"

    # get time at training
    registered_time = datetime.now().strftime("%d/%m/%Y %H:%M:%S")


writer = SummaryWriter(log_dir="SummaryWriter_log", filename_suffix="wb")
def train_model(model, option, SOURCE, TARGET):
    print("Training model...")
    model.train()

    with open(log_train_path, 'a') as f:
        f.write(f'Train at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}\n')
    f1_metric = F1Score(ignore_index=option.target_pad).to(option.cuda_device)

    for epoch in range(option.epochs):
        start_time = time.time()

        for iteration, batch in tqdm(enumerate(option.train), total=50, colour="CYAN"):
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

            if option.lr_scheduler == True:
                option.scheduler.step()

        with open(log_train_path, 'a') as f:
            f.write(f"Epoch {epoch + 1}/{option.epochs}\n")
            f.write(f"Time: {time.time() - start_time}.\n")
            f.write(f"Loss: {loss.item()}\n")
            if option.checkpoint == True:
                f.write(f"F1 Score: {f1_score}\n")
            else:
                f.write(f"F1 Score: {f1_score}\n\n")

        print((f"Epoch {epoch + 1}/{option.epochs}:"))
        print(f"Time: {time.time() - start_time}.")
        print(f"Loss: {loss.item()}")
        print(f"F1 Score: {f1_score}")

        # write to SummaryWriter()
        writer.add_scalar('Epoch', epoch + 1, epoch)
        writer.add_scalar('Time', time.time() - start_time, epoch)
        writer.add_scalar('Loss', loss.item(), epoch)
        writer.add_scalar('F1 score', f1_score, epoch)

        if option.checkpoint == True:
            with open(log_train_path, 'a') as f:
               f.write(f"Save model after {epoch + 1} epoch(s).\n\n")

            torch.save(model.state_dict(), model_path)
            print(f"Save model after {epoch + 1} epoch(s).")

        option.train = create_data(option, SOURCE, TARGET, repeat=1)
    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=bool, default=False)
    parser.add_argument('-data_file', type=str, default=data_path)
    parser.add_argument('-source_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-target_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-epochs', type=int, default=15)
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=bool, default=cuda)
    parser.add_argument('-cuda_device', type=str, default=processor)
    parser.add_argument('-batch_size', type=int, default=10)
    parser.add_argument('-lr', type=int, default=0.0001)
    parser.add_argument('-threshold', type=int, default=0.9)
    parser.add_argument('-max_strlen', type=int, default=300)
    parser.add_argument('-checkpoint', type=bool, default=True)
    parser.add_argument('-lr_scheduler', type=bool, default=True)

    option = parser.parse_args()

    SOURCE, TARGET = create_files(option)
    option.train = create_data(option, SOURCE, TARGET)

    print(f'Length of SOURCE vocab: {len(SOURCE.vocab)}, TARGET vocab: {len(TARGET.vocab)}')
    model = get_model(option, len(SOURCE.vocab), len(TARGET.vocab))
    option.optimizer = torch.optim.Adam(params=model.parameters(), lr=option.lr, betas=(0.9, 0.9999), eps=1e-9, weight_decay=0)

    if option.lr_scheduler == True:
        option.scheduler = CosineWithWarmRestarts(option.optimizer, T_max=option.train_len)

    if option.checkpoint == True:
        print("Model is saved at the end of each epoch.")

    train_model(model, option, SOURCE, TARGET)


if __name__ == "__main__":
    main()



