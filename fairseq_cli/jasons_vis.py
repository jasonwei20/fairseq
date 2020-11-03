import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def plot_jasons_lineplot(
    x_list,
    y_list_list,
    y_labels_list, 
    x_ax_label,
    y_ax_label,
    title,
    output_png_path,
    ):
    
    if x_list == None:
        x_list = range(1, len(y_list_list[0]) + 1)

    _, ax = plt.subplots()

    for y_list, y_label in zip(y_list_list, y_labels_list):
        plt.plot(x_list, y_list, label=y_label, linewidth=1)

    plt.xlabel(x_ax_label)
    plt.ylabel(y_ax_label)
    plt.title(title)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_png_path, dpi=400)
    plt.clf()

    print(f"plot saved at {output_png_path}")
