import matplotlib.pyplot as plt
def plot_metrics_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Lấy chỉ số, loss và F1 score từ file
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

    # Biểu đồ loss
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Biểu đồ F1 score
    plt.subplot(1, 2, 2)
    plt.plot(epochs, f1_scores, label='Training F1 Score', color='orange')
    plt.title('Training F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()

    # Hiển thị biểu đồ
    plt.tight_layout()
    plt.show()

# Sử dụng hàm để vẽ biểu đồ từ file
file_path = 'weights/log_train.txt'  # Điều chỉnh đường dẫn đến file của bạn
plot_metrics_from_file(file_path)
