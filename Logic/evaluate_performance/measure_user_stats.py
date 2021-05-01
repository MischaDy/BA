import time

from Logic.ProperLogic.main_logic import init_program, get_user_command, call_handler

from Logic.ProperLogic.commands import Command, Commands
from Logic.ProperLogic.database_modules.database_logic import DBManager


EMBEDDINGS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'


def run_program_with_user_stats():
    write = False
    command_stats_path = r'C:\Users\Mischa\Desktop\Uni\20-21 WS\Bachelor\BA Papers\Datasets\faces 1999 caltech\commands_stats.txt'
    t0 = time.time()

    # Models.altered_mtcnn.keep_all = False
    init_program()
    cluster_dict = DBManager.load_cluster_dict()

    commands = []
    cmd_name = get_user_command()
    while cmd_name != str(Commands.exit):
        t1 = time.time()
        cmd = Command.get_command(cmd_name)
        call_handler(cmd.handler, cluster_dict=cluster_dict)
        t2 = time.time()
        commands.append([cmd_name, t2 - t1])
        cmd_name = get_user_command()

    tn = time.time()
    commands_str = '\n'.join(map(str, commands)) + '\n\n' + f'total runtime: {tn - t0}'
    if write:
        with open(command_stats_path, 'w') as file:
            file.write(commands_str)


if __name__ == '__main__':
    run_program_with_user_stats()
