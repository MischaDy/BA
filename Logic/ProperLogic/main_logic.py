"""
Program containing the main application logic.
"""

import os

import torchvision
from PIL import Image

import input_output_logic


# TODO: 'clean input' function, with lower and strip


#TODO: consistent paths!
TENSORS_PATH = 'Logic/ProperLogic/stored_embeddings'
CLUSTERS_PATH = 'stored_clusters'

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'
TO_TENSOR = torchvision.transforms.ToTensor()

TERMINATING_TOKENS = ('halt', 'stop', 'quit', 'exit',)
# TODO: add 'help' command
COMMANDS_DESCRIPTIONS = {
    'select new faces': 'add',
    'edit existing faces': 'edit',
    'find individual': 'find',
    'reclassify individuals': 'reclassify',
    'show cluster': 'showcluster',
}

HANDLERS = {
    'showcluster': handler_showcluster,
}

# TODO: How to draw + store boundaries of clusters?
#       --> "Cluster-Voronoi-Diagram"?? Spheres / specific ellipsoids of bounded size? Does this generalize well to many
#           dimensions?
# TODO: What should / shouldn't be private?
# TODO: Turn Commands into an Enum
# TODO: Associate Handler with each command
# TODO: Consistent naming
# TODO: Add comments & docstring


def main(handlers, terminating_tokes, clusters_path):
    handlers_params = {
        'showcluster': (clusters_path),
    }

    command = ''
    while command not in terminating_tokes:
        command = get_user_command()
        process_command(command, handlers, handlers_params)


def process_command(command, handlers, handlers_params):
    # TODO: complete function
    # if command in 'add':  # select new faces
    #     # TODO: complete section
    #     add_new_embeddings()
    #
    # elif command in 'edit':  # edit existing faces (or rather, existing identities)
    #     ...
    #
    # elif command in 'find':
    #     ...

    try:
        handler = handlers[command]
        params = handlers_params[command]
        handler(*params)  # seems to work if params is empty, as well

    except KeyError:
        print('Error, unknown command.')
        # raise NotImplementedError(f"command '{command}'")


# ----- I/O -----

def get_user_command():
    command = _get_user_command_subfunc()
    while command not in COMMANDS_DESCRIPTIONS.values():
        print_error_msg('Unknown command, please try again.', False)
        command = _get_user_command_subfunc()
    return command


def _get_user_command_subfunc():
    print('\nWhat would you like to do next?')
    print_command_options()
    return input().lower().strip()


def print_command_options():
    cmd_options_lines = (f"- To {command}, type '{abbreviation}'." for command, abbreviation in COMMANDS_DESCRIPTIONS.items())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)


# --- i/o helpers ---


def _choose_cluster(clusters_path):
    clusters_names = list(get_clusters_gen(clusters_path))
    print_clusters(clusters_names)
    chosen_cluster = input()
    while chosen_cluster not in clusters_names:
        print(f'Error, cluster {chosen_cluster} not found. Please try again.')
        print_clusters(clusters_names)
        chosen_cluster = input()
    return chosen_cluster


def print_clusters(clusters):
    TEMP_LIM = 10
    # TODO: print clusters limited number at a time (Enter=continue
    clusters_names = clusters[:TEMP_LIM]  # TODO: remove this line
    clusters_str = '\n'.join(map(lambda string: f'- {string}', clusters_names))
    print('Which cluster would you like to view?')
    print(clusters_str)


# ----- FILE I/O -----

def get_clusters_gen(clusters_path):
    return (string for string in os.listdir(clusters_path) if os.path.isdir(string))


# ----- COMMAND PROCESSING -----

def handler_showcluster(clusters_path):
    choose_another_cluster = ''
    while 'n' not in choose_another_cluster:
        chosen_cluster = _choose_cluster(clusters_path)



        choose_another_cluster = input('Choose another cluster?\n').lower().strip()





def add_new_embeddings():
    """
    1. User selects new images to extract faces out of
    2. Extract faces
    3. Store embeddings in vector space


    """
    # Img Selection + Face Extraction
    face_embeddings_gen = user_choose_imgs()
    add_to_embeddings_to_vector_space(face_embeddings_gen)


def add_to_embeddings_to_vector_space(embeddings):
    """

    :param embeddings: An iterable containing the embeddings
    :return:
    """
    pass


def user_choose_imgs():
    # TODO: Implement
    path = user_choose_path()
    extract_faces(path)
    face_embeddings_gen = load_img_tensors_from_dir(path)
    return face_embeddings_gen


def user_choose_path():
    # TODO: Implement
    return IMG_PATH


def extract_faces(path):
    # TODO: Implement
    return 0


def load_img_tensors_from_dir(dir_path, img_extensions=None, output_file_name=False):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param output_file_name: Whether the tensor should be yielded together with the corresponding file name
    :param dir_path: Directory containing images
    :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}. Empty = no
    filtering
    :return: Yield(!) tuples of image_names and tensors contained in this folder
    """
    # TODO: Finish implementing
    if img_extensions is None:
        img_extensions = set()
    if not output_file_name:
        for img_name in get_img_names(dir_path, img_extensions):
            with Image.open(dir_path + os.path.sep + img_name) as img:
                yield _to_tensor(img)
    else:
        for img_name in get_img_names(dir_path, img_extensions):
            with Image.open(dir_path + os.path.sep + img_name) as img:
                yield img_name, _to_tensor(img)


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


def get_img_names(dir_path, recursive=False, img_extensions=None):
    # TODO: Finish implementing
    # TODO: Implement recursive option
    # TODO: Implement extension filtering
    if img_extensions is None:
        img_extensions = set()
    return list(filter(lambda name: os.path.isfile(name), os.listdir(dir_path)))


# ------- HELPER FUNCTIONS ------- #

def print_error_msg(msg, print_newline=True):
    print('\n' + msg, end='\n' if print_newline else '')


if __name__ == '__main__':
    main(HANDLERS, TERMINATING_TOKENS, CLUSTERS_PATH)
