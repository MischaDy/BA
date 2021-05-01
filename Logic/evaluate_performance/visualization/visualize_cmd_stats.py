import os
from matplotlib import pyplot as plt

from Logic.ProperLogic.misc_helpers import get_every_nth_item, get_ext

RESULTS_DIR_PATH = '../results_cmd_stats'
CMD_STATS_FILE = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech\commands_stats.txt'


def main():
    with open(CMD_STATS_FILE, 'r') as file:
        lines = file.readlines()

    cmd_dict = {'process_images_dir': [], 'reclassify': [], 'clear_data': []}
    for line in lines:
        cmd_name, n, duration = line.strip('[]\n').split(', ')
        cmd_name, n, duration = cmd_name.strip(" '"), int(n), float(duration)
        cmd_dict[cmd_name].append((n, round(duration, 3)))

    for cmd_name, xs_and_ys in cmd_dict.items():
        xs, ys = list(get_every_nth_item(xs_and_ys, 0)), list(get_every_nth_item(xs_and_ys, 1))
        save_path = os.path.join(RESULTS_DIR_PATH, cmd_name + '.png')
        plot_cmd(cmd_name, xs, ys, save_path)


def plot_cmd(cmd_name, xs, ys, save_path=None):
    x_eps = 450 / 20
    y_eps = max(ys) / 20
    x_axis_limits = [0 - x_eps, 450 + x_eps]
    y_axis_limits = [0 - y_eps, max(ys) + y_eps]

    fig, ax = plt.subplots()
    plt.plot(xs, ys)
    plt.title(cmd_name)
    ax.set_xlim(x_axis_limits)
    ax.set_ylim(y_axis_limits)
    ax.set_xlabel('number of images processed')
    ax.set_ylabel('runtime in seconds')

    if save_path is not None:
        plt.savefig(save_path, format=get_ext(save_path))
    plt.show()


if __name__ == '__main__':
    main()
