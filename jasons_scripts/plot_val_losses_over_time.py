import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def find_y_at_lowest_x(x_list, y_list):
    x_min = min(x_list)
    convergence_index = x_list.index(x_min)
    y_at_x_min = y_list[convergence_index]
    return x_min, y_at_x_min, convergence_index

def plot_many_lineplot(
    x_list_list,
    y_list_list,
    y_labels_list, 
    x_ax_label,
    y_ax_label,
    title,
    output_png_path,
    ):

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
    colors = colors[:len(x_list_list)]

    _, ax = plt.subplots()

    for x_list, y_list, y_label, color in zip(x_list_list, y_list_list, y_labels_list, colors):
        x_min, y_at_x_min, convergence_index = find_y_at_lowest_x(x_list, y_list)
        # print(f"for {y_label}, best nll={x_min:.3f} and uid loss at best nll={y_at_x_min:.3f}")
        # plt.plot(x_min, y_at_x_min, color + 'x', markersize=10, label=y_label)
        plt.plot(x_list, y_list, color, linewidth=0.3, label=y_label)

    plt.xlabel(x_ax_label)
    plt.ylabel(y_ax_label)
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=400)
    plt.clf()

    print(f"plot saved at {output_png_path}")

def read_file(file_path):
    data = pd.read_csv(file_path)
    updates = list(data['updates'])[:300]
    val_loss = list(data['val_loss'])[:300]
    val_uid_loss = list(data['val_uid_loss'])#[36:270]

    return updates, val_loss

def plot_val_losses_time(file_path_list, y_labels_list, output_png_path):

    baseline_updates, baseline_loss = read_file(file_path_list[0])
    _, var_loss = read_file(file_path_list[1])
    loss_diff = [var_loss[i] - baseline_loss[i] for i in range(len(baseline_updates))]
    x_list_list = [baseline_updates]
    y_list_list = [loss_diff]

    # x_list_list = []
    # y_list_list = []
    # for file_path in file_path_list:
    #     updates, val_loss = read_file(file_path)
    #     x_list_list.append(updates)
    #     y_list_list.append(val_loss)
    
    plot_many_lineplot(
        x_list_list = x_list_list,
        y_list_list = y_list_list,
        y_labels_list = ['val loss diff: uid regularized - baseline'],
        x_ax_label = 'updates',
        y_ax_label = 'val loss',
        title = '',
        output_png_path = output_png_path,
    )

if __name__ == "__main__":

    plot_val_losses_time(
        file_path_list = [
            'jason-lm-logs-ptb/dropout03_allvar00/train_logs_backup.csv',
            'jason-lm-logs-ptb/dropout03_allvar01/train_logs_backup.csv',
        ],
        y_labels_list = [
            'variance 0.0',
            'variance 0.03',
        ],
        output_png_path = 'jason-plots/variance_valloss_overtime.png'
    )
