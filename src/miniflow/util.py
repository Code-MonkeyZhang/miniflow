import copy
import numpy as np
import sys
import matplotlib.pyplot as plt



def plot_loss(epoch_lost_list):
    # 创建一个新的图形
    plt.figure()

    # 绘制数据线
    plt.plot(epoch_lost_list)

    # 在数据点上绘制点
    plt.scatter(range(len(epoch_lost_list)), epoch_lost_list, color='red', marker='o', s=10)

    # 添加标题和标签
    plt.title('Data Plot with Points')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # 显示图形
    plt.show()


def train_summary(loss, time):
    """
    Generates a concise summary of the training process.

    Args:
    loss (list): List of loss values for each epoch.
    time (list): List of training times for each epoch (in seconds).

    Returns:
    None: Prints the summary to the console.
    """
    total_time = sum(time)
    avg_epoch_time = sum(time) / len(time) if time else 0
    final_loss = loss[-1] if loss else None
    min_loss = min(loss) if loss else None
    epochs = len(loss)

    print("TRAINING SUMMARY")
    print("=" * 40)
    print(f"Epochs: {epochs}")
    print(f"Total Time: {total_time:.0f} ms")
    print(f"Avg Time/Epoch: {avg_epoch_time:.1f} ms")
    print(f"Final Loss: {final_loss:.4f}" if final_loss is not None else "Final Loss: N/A")
    print(f"Best Loss: {min_loss:.4f}" if min_loss is not None else "Best Loss: N/A")
    print("=" * 40)


def compute_linear_gradient(x_train, y_train, w, b, lambda_=0):
    size = x_train.shape[0]
    features = x_train.shape[1]
    dj_dw = np.zeros(features)
    dj_db = 0

    # 遍历每一行数据
    for i in range(size):
        f_wb = linear_function(x_train[i], w, b)
        error = (f_wb - y_train[i])
        # 遍历每一个feature
        for j in range(features):
            dj_dw[j] += error * x_train[i, j]  # 计算每个feature的导数，然后累加起来，方便后面求平均
        # 对于 b, 只需要算一次
        dj_db += error

    dj_dw = dj_dw / size
    dj_db = dj_db / size

    for j in range(features):
        dj_dw[j] += lambda_ * w[j] / size

    return dj_dw, dj_db

# feature scaling
def feature_scaling(data: np.ndarray, type: str) -> np.ndarray:
    if type == 'z-score':
        # Z-Score Normalization
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        normalized_data = (data - mean) / std
    elif type == 'min-max':
        # Min-Max Normalization
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - min_val) / (max_val - min_val)
    elif type == 'mean':
        # Mean Normalization
        mean = np.mean(data, axis=0)
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        normalized_data = (data - mean) / (max_val - min_val)
    else:
        raise ValueError("Type must be 'z-score', 'min-max', or 'mean'")

    return normalized_data


def print_progress_bar(index, count, bar_length=50):
    """
    Prints a progress bar in the console.
    Args:
        index (int): The current index value representing the progress made.
        count (int): The total count or number of items to be processed.
        bar_length (int, optional): The length of the progress bar. Defaults to 50.
    Example:
        >>> print_progress_bar(10, 100, bar_length=30)
        [=========>                             ]
    """

    # Calculate the completion percentage
    percent_complete = float(index) / count
    # Calculate the length of the completed portion
    filled_length = int(round(bar_length * percent_complete))
    # Create the progress bar character representation
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)
    # Format and output the progress bar
    sys.stdout.write('\r[%s]' % bar)
    sys.stdout.flush()


def slice2batches(X_train: np.ndarray, y_train: np.ndarray, batch_size: int):
    """
    Given the entire training set & its label, cut them into pieces with the size of batch size
    Args:
        X_train (numpy): The training set
        y_train (numpy): The corresponding label set
        batch_size (int): size of each piece
    Example:
        X_train: (64,28,28)
        y_train: (64,1)
        batch_size: 32

        train_batch_list, label_batch_list = slice2batches(X_train, y_train, batch_size):

        X_batch_list: [(32,28,28), (32,28,28)]
        y_batch_list: [(32,1),(32,1)]
    """
    num_batches = (len(X_train) + batch_size - 1) // batch_size
    X_batch_list = np.array_split(X_train, num_batches)
    y_batch_list = np.array_split(y_train, num_batches)

    return X_batch_list, y_batch_list


def compute_cross_entropy_loss(prediction, target):
    epsilon = 1e-12
    prediction = np.clip(prediction, epsilon, 1. - epsilon)
    lost_per_sample = -np.sum(target * np.log(prediction), axis=1)
    lost = np.mean(lost_per_sample)
    return lost


def label2onehot(label, units):
    label_one_hot = np.zeros((label.shape[0], units))
    label_one_hot[np.arange(label.shape[0]), label] = 1
    return label_one_hot


def conv_single_step(image_slice, filter_weights):
    """
    Simple Convolution operation for a single step.
    It multiplies a_slice_prev and filter_weights, and then sums over all entries.
    Basic building block for a convolutional layer.
    """
    # Element-wise product between a_slice_prev and W. Do not add the bias yet.
    s = np.multiply(image_slice, filter_weights)
    # Sum over all entries of the volume s.
    Z = np.sum(s)

    return Z
