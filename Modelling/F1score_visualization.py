from matplotlib import pyplot as plt


def plot_metrics_from_file(file_path):
    with open(file_path, 'r') as file:
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


def main() -> None:
    file_path = "weights/log_train.txt"
    plot_metrics_from_file(file_path)
    return None


if __name__ == '__main__':
    main()
