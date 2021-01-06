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
        print(f"for {y_label}, best nll={x_min:.3f} and uid loss at best nll={y_at_x_min:.3f}")
        plt.plot(x_min, y_at_x_min, color + 'x', markersize=10, label=y_label)
        plt.plot(x_list, y_list, color, linewidth=0.5, label=y_label)

    plt.xlabel(x_ax_label)
    plt.ylabel(y_ax_label)
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=400)
    plt.clf()

    print(f"plot saved at {output_png_path}")

def read_file(file_path, graph_type):
    data = pd.read_csv(file_path)
    updates = list(data['updates'])[:200]
    val_loss = list(data['val_loss'])[:200]
    val_uid_loss = list(data['val_uid_loss'])[:200]

    if graph_type == 'nll-uid':
        return val_loss, val_uid_loss
    elif graph_type == 'updates-uid':
        return updates, val_uid_loss

def plot_nll_uid(
    file_path_list,
    y_labels_list,
    y_ax_label,
    output_png_path,
    graph_type  = 'nll-uid',
    x_ax_label = 'nll loss on dev set',
    ):

    x_list_list = []
    y_list_list = []
    for file_path in file_path_list:
        val_loss, val_uid_loss = read_file(file_path, graph_type)
        x_list_list.append(val_loss)
        y_list_list.append(val_uid_loss)
    
    plot_many_lineplot(
        x_list_list = x_list_list,
        y_list_list = y_list_list,
        y_labels_list = y_labels_list,
        x_ax_label = x_ax_label,
        y_ax_label = y_ax_label,
        title = '',
        output_png_path = output_png_path,
    )

if __name__ == "__main__":
    # plot_nll_uid(
    #     file_path_list = [
    #         'jason-lm-logs-wt2/withuid_dropout00_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout00_allvar003/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar005/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout02_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout02_allvar003/train_logs_backup.csv',
    #     ],
    #     y_labels_list = [
    #         'dropout 0.0', 
    #         'dropout 0.0 + var uid 0.03', 
    #         'dropout 0.1', 
    #         'dropout 0.1 + var uid 0.05', 
    #         'dropout 0.2',
    #         'dropout 0.2 + var uid 0.03'
    #     ],
    #     y_ax_label = 'uid loss: variance on dev set',
    #     output_png_path = 'jason-plots/var_nll_plot.png'
    # )

    # comparing dropout
    # plot_nll_uid(
    #     file_path_list = [
    #         'jason-lm-logs-wt2/withuid_dropout00_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout005_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout0075_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout015_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout02_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout025_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout03_allvar00/train_logs_backup.csv',
    #     ],
    #     y_labels_list = [
    #         'dropout 0.0', 
    #         'dropout 0.05',
    #         'dropout 0.075',
    #         'dropout 0.1', 
    #         'dropout 0.15', 
    #         'dropout 0.2',
    #         'dropout 0.25',
    #         'dropout 0.3',
    #     ],
    #     y_ax_label = 'uid loss: variance on dev set',
    #     output_png_path = 'jason-plots/var_nll_plot.png',
    #     # x_ax_label = 'updates',
    #     # graph_type = 'updates-uid',
    # )
    
    #comparing UID beta
    # plot_nll_uid(
    #     file_path_list = [
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar00/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar001/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar003/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar005/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar006/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar007/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar008/train_logs_backup.csv',
    #         'jason-lm-logs-wt2/withuid_dropout01_allvar01/train_logs_backup.csv',
    #     ],
    #     y_labels_list = [
    #         'dropout 0.1 + uid var reg 0.0', 
    #         'dropout 0.1 + uid var reg 0.01', 
    #         'dropout 0.1 + uid var reg 0.03', 
    #         'dropout 0.1 + uid var reg 0.05', 
    #         'dropout 0.1 + uid var reg 0.06', 
    #         'dropout 0.1 + uid var reg 0.07', 
    #         'dropout 0.1 + uid var reg 0.08', 
    #         'dropout 0.1 + uid var reg 0.1', 
    #     ],
    #     y_ax_label = 'uid loss: variance on dev set',
    #     output_png_path = 'jason-plots/var_nll_plot_beta.png'
    # )

    #new
    plot_nll_uid(
        file_path_list = [
            'jason-lm-logs-europv7-en/allvar00/train_logs_backup.csv',
            'jason-lm-logs-europv7-en/allvar001/train_logs_backup.csv',
            'jason-lm-logs-europv7-en/allvar002/train_logs_backup.csv',
            'jason-lm-logs-europv7-en/allvar003/train_logs_backup.csv',
        ],
        y_labels_list = [
            'uid var reg 0.0', 
            'uid var reg 0.01', 
            'uid var reg 0.02', 
            'uid var reg 0.03', 
        ],
        y_ax_label = 'uid loss: variance on dev set',
        output_png_path = 'jason-plots/europv7-en-test.png'
    )