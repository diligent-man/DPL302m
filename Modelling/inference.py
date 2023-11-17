import os
import time
from pandas._libs import interval
import torch
import argparse
import multiprocessing as mp
from transformer import get_model
from beam_search import beam_search
from torch.autograd import Variable
from preprocess import create_files, preprocess


# check gpu & time
if torch.cuda.is_available():
    cuda = True
    processor = "cuda"
else:
    cuda = False
    processor = "cpu"


class Inferer():
    def __init__(self) -> None:
        __parser = argparse.ArgumentParser()
        __parser.add_argument('-load_weights', type=bool, default=True)
        __parser.add_argument('-k', type=int, default=1)
        __parser.add_argument('-source_lang', type=str, default="en_core_web_sm")
        __parser.add_argument('-target_lang', type=str, default="en_core_web_sm")
        __parser.add_argument('-d_model', type=int, default=512)
        __parser.add_argument('-n_layers', type=int, default=6)
        __parser.add_argument('-heads', type=int, default=8)
        __parser.add_argument('-dropout', type=int, default=0.1)
        __parser.add_argument('-cuda', type=bool, default=cuda)
        __parser.add_argument('-cuda_device', type=str, default=processor)
        __parser.add_argument('-max_strlen', type=int, default=300)

        self.__option = __parser.parse_args()
        self.__SOURCE, self.__TARGET = create_files(self.__option)
        self.__model = get_model(self.__option, len(self.__SOURCE.vocab), len(self.__TARGET.vocab))

    @staticmethod
    def __process_sentence(sentence, model, option, SOURCE, TARGET):
        model.eval()  # equivilent with

        sentence = preprocess(sentence)
        indexed = []

        for token in sentence:
            indexed.append(SOURCE.vocab.stoi[token])

        sentence = Variable(torch.LongTensor([indexed]))
        sentence = beam_search(sentence, model, SOURCE, TARGET, option)
        return sentence


    def __process(self) -> list:
        corrected_sentences = []
        for sentence in self.__option.text:
            # .capitalize()): convert the first letter to uppercase letter
            corrected_sentences.append(
                self.__process_sentence(sentence, self.__model, self.__option, self.__SOURCE, self.__TARGET).capitalize()
            )
        return corrected_sentences


    def infer_from_file(self, filename) -> None:
        if os.path.exists("output.txt"):
            os.remove("output.txt")

        sentences = []
        with open(file=filename, mode="r", encoding='utf-8') as f:
            for text in f.readlines():
                sentences.append(text[:-1])  # skip endline character
        self.__option.text = sentences
        print("Processing...")
        correct_sentences = self.__process()

        with open("output.txt", "w") as f:
            for sentence in correct_sentences:
                f.write(sentence + "\n")
        print("Finished")
        return None


    def infer_from_text(self, text: str) -> str:
        self.__option.text = [sentence.strip() for sentence in text.split(".") if len(sentence) != 0]
        
        print("Processing...")
        corrected_sentences = ". ".join(self.__process()) + "."
        print("Finished")
        return corrected_sentences


def main():
    inferer = Inferer()
    with open("input.txt") as f:
        data = f.readlines()
    data = [i for i in data]

    for text in data:
        start = time.time()
        inferer.infer_from_text(text)
        print(time.time() - start)


if __name__ == '__main__':
    main()
