import numpy as np
from matplotlib import pyplot as plt


def log_train_plotting(filepath):
    with open(filepath, 'r') as file:
        lines = file.readlines()

    epochs = []
    losses = []
    f1_scores = []

    for line in lines:
        if 'Epoch' in line:
            epoch = int(line.split('/')[0].split(' ')[-1])
            epochs.append(epoch)
        elif 'Loss' in line:
            loss = float(line.split(':')[-1].strip())
            losses.append(loss)
        elif 'F1 Score' in line:
            f1_score = float(line.split(':')[-1].strip())
            f1_scores.append(f1_score)

    # Loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim([0, max(epochs)+10])
    plt.xticks(np.arange(0, max(epochs), 10))
    plt.legend(loc='upper right')

    # F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores, label='F1 Score', color='orange')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend(loc='upper left')

    # Hiển thị biểu đồ
    plt.tight_layout()
    plt.show()


def processing_time_plotting(filepath) -> None:
    # Data
    processors = ['Nvidia T4', 'Nvidia V100']
    word_count = ['100 words text', '200 words text', '300 words text', '500 words text']

    # Processing times in seconds for each processor and word count
    nvidia_t4_times = [18.37563943862915, 31.50808095932007, 48.04881525039673, 78.56428670883179]
    nvidia_v100_times = [12.372150659561157, 23.5001540184021, 34.505385875701904, 58.1375253200531]

    x = range(len(word_count))  # the label locations

    # Plotting the bars
    bar_width = 0.2
    plt.figure(figsize=(10, 6))

    plt.bar(x, nvidia_t4_times, width=bar_width, label='Nvidia T4')
    plt.bar([i + bar_width for i in x], nvidia_v100_times, width=bar_width, label='Nvidia V100')

    plt.ylim(0, 100)
    # Adding labels and title
    plt.xlabel('Word Count')
    plt.ylabel('Processing Time (seconds)')
    plt.title('Processing Time by Processor and Word Count')
    plt.xticks(x, word_count)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.xticks(rotation=0)  # Rotate x labels for better readability
    plt.show()
    return None


def main() -> None:
    filepath = "weights/log_train.txt"
    # log_train_plotting(file_path)

    filepath = "processing_time.txt"
    processing_time_plotting(filepath)
    return None


if __name__ == '__main__':
    main()
