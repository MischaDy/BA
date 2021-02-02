"""
Program containing the main application logic.
"""

import os

import torchvision
from PIL import Image

#from abc import ABC

import input_output_logic


# TODO: Always allow option to leave current menu item / loop rather than continue!


# TODO: consistent paths!
from Logic.misc_helpers import clean_str

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
    'show a cluster': 'showcluster',
}


def get_handlers_dict():
    return {
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


# class Handler:
#     def __init__(self):
#         pass


def main(terminating_tokes, clusters_path):
    handlers = get_handlers_dict()
    handlers_params = {
        'showcluster': [clusters_path],
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
        print_error_msg('unknown command')
        # raise NotImplementedError(f"command '{command}'")


# ----- I/O -----

def get_user_command():
    # TODO: make user choose command
    command = 'showcluster'  # _get_user_command_subfunc()
    while command not in COMMANDS_DESCRIPTIONS.values():
        print_error_msg('Unknown command, please try again.', False)
        command = _get_user_command_subfunc()
    return command


def _get_user_command_subfunc():
    _wait_for_any_input('What would you like to do next?')  # TODO: needs \n?
    print_command_options()
    return clean_str(input())


def print_command_options():
    cmd_options_lines = (f"- To {command}, type '{abbreviation}'."
                         for command, abbreviation in COMMANDS_DESCRIPTIONS.items())
    output = '\n'.join(cmd_options_lines) + '\n'
    print(output)


def _output_cluster_content(cluster_name, cluster_path):
    _wait_for_any_input(f'Which face image in the cluster "{cluster_name}" would you like to view?')
    # TODO: finish; output faces and (-> separate function?) allow choice of image


# --- i/o helpers ---


def _user_choose_cluster(clusters_path, return_names=True):
    clusters_names_and_paths = list(get_clusters_gen(clusters_path, return_names=True))
    clusters_names = _get_nth_tuple_elem(clusters_names_and_paths, n=0)
    prompt_cluster_choice(clusters_names)
    chosen_cluster_name = input()
    while chosen_cluster_name not in clusters_names:
        print_error_msg(f'cluster "{chosen_cluster_name}" not found; Please try again.')
        prompt_cluster_choice(clusters_names)
        chosen_cluster_name = input()

    chosen_cluster_path = next(filter(lambda iterable: iterable[0] == chosen_cluster_name, clusters_names_and_paths))[1]
    if return_names:
        return chosen_cluster_name, chosen_cluster_path
    return chosen_cluster_path


def _get_nth_tuple_elem(iterables, n=0):
    """
    Return nth element (zero-indexed!) in each iterable stored in the iterable.

    Example: _get_nth_tuple_elem(zip(range(3, 7), 'abcdefgh'), n=1) --> ['a', 'b', 'c', 'd']

    iterables: iterable of indexable iterables, each of at least length n-1 (since n is an index).
    n: index of element to return from each stored iterable
    """
    return list(map(lambda iterable: iterable[n], iterables))


def prompt_cluster_choice(clusters_names):
    TEMP_LIM = 10
    # TODO: print clusters limited number at a time (Enter=continue)
    clusters_names = clusters_names[:TEMP_LIM]  # TODO: remove this line
    clusters_str = '\n'.join(map(lambda string: f'- {string}', clusters_names))
    _wait_for_any_input('Which cluster would you like to view? (Press any key to continue.)')
    print(clusters_str)


def _wait_for_any_input(prompt):
    input(prompt + '\n')


# ----- FILE I/O -----

def get_clusters_gen(clusters_path, return_names=True):
    file_obj_names = os.listdir(clusters_path)
    file_obj_paths = map(lambda obj_name: os.path.join(clusters_path, obj_name), file_obj_names)
    clusters_names_and_paths = filter(lambda obj_tup: os.path.isdir(obj_tup[1]), zip(file_obj_names, file_obj_paths))
    if return_names:
        return clusters_names_and_paths
    # return only paths
    return _get_nth_tuple_elem(clusters_names_and_paths, n=1)


# ----- COMMAND PROCESSING -----

def handler_showcluster(clusters_path):
    should_continue = ''
    while 'n' not in should_continue:
        cluster_name, cluster_path = _user_choose_cluster(clusters_path)
        _output_cluster_content(cluster_path)




        should_continue = clean_str(input('Choose another cluster?\n'))


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
    print('\nError: ' + msg, end='\n' if print_newline else '')


if __name__ == '__main__':
    main(TERMINATING_TOKENS, CLUSTERS_PATH)
