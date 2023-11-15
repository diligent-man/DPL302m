import argparse
import torch
from transformer import get_model
from preprocess import *
from beam_search import beam_search
from torch.autograd import Variable

def process_sentence(sentence, model, option, SOURCE, TARGET):
    model.eval()
    indexed = []
    sentence = preprocess(sentence)
    for tok in sentence:
        indexed.append(SOURCE.vocab.stoi[tok])
    sentence = Variable(torch.LongTensor([indexed]))
    sentence = beam_search(sentence, model, SOURCE, TARGET, option)
    return sentence

def process(option, model, SOURCE, TARGET):
    sentences = option.text
    correct_sentences = []
    for sentence in sentences:
        correct_sentences.append(process_sentence(sentence, model, option, SOURCE, TARGET).capitalize()) 
        # .capitalize()): convert the first letter to uppercase letter
    return correct_sentences

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-load_weights', type=bool, default=True)
    parser.add_argument('-k', type=int, default=3)
    parser.add_argument('-source_language', type=str, default="en_core_web_sm")
    parser.add_argument('-target_lang', type=str, default="en_core_web_sm")
    parser.add_argument('-d_model', type=int, default=512)
    parser.add_argument('-n_layers', type=int, default=6)
    parser.add_argument('-heads', type=int, default=8)
    parser.add_argument('-dropout', type=int, default=0.1)
    parser.add_argument('-cuda', type=bool, default=False)
    parser.add_argument('-cuda_device', type=str, default="cpu")
    parser.add_argument('-max_strlen', type=int, default=300)

    option = parser.parse_args()

    SOURCE, TARGET = create_files(option)
    model = get_model(option, len(SOURCE.vocab), len(TARGET.vocab))
    
    while True:
        option.text = input("Enter a filename to process (type \"quit\" to escape):\n")
        if option.text == "quit":
            break
        
        sentences = []
        with open(option.text, "r", encoding='utf-8') as f:
            for text in f.readlines():
                sentences.append(text[:-1]) # skip endline character
        option.text = sentences
        print("Processing...")
        correct_sentences = process(option, model, SOURCE, TARGET)
        if os.path.exists("output.txt"):
            os.remove("output.txt")
        with open("output.txt","w") as f:
            for sentence in correct_sentences:
                f.write(sentence + "\n")
        f.close()
        print("Finished.")
        # except:
        #     print("Error: Cannot open text file.")
        #     continue

if __name__ == '__main__':
    main()
