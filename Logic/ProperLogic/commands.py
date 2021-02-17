import os

import torchvision
from PIL import Image

from Logic.misc_helpers import log_error, clean_str, wait_for_any_input, get_nth_tuple_elem

IMG_PATH = 'Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive'

TO_TENSOR = torchvision.transforms.ToTensor()


class Command:
    # TODO: add 'help' command
    commands = {}

    def __init__(self, cmd_name, cmd_desc, handler, handler_params=None):
        self.cmd_name = cmd_name
        self.cmd_desc = cmd_desc
        self.handler = handler
        if handler_params is None:
            handler_params = []
        self.handler_params = handler_params
        type(self).commands[self.cmd_name] = self

    def get_cmd_name(self):
        return self.cmd_name

    def set_cmd_name(self, cmd_name):
        self.cmd_name = cmd_name

    def get_cmd_description(self):
        return self.cmd_desc

    def set_cmd_description(self, new_cmd_desc):
        self.cmd_desc = new_cmd_desc

    def get_handler_params(self):
        return self.handler_params

    def set_handler_params(self, new_handler_params):
        self.handler_params = new_handler_params

    @classmethod
    def get_commands(cls):
        return cls.commands

    @classmethod
    def get_command_names(cls):
        return cls.commands.keys()

    @classmethod
    def get_command_descriptions(cls):
        return [(cmd.cmd_name, cmd.cmd_desc) for cmd in cls.commands]

    @classmethod
    def remove_command(cls, cmd_name):
        try:
            cls.commands.pop(cmd_name)
        except KeyError:
            log_error(f"Could not remove unknown command '{cmd_name}'")

    @classmethod
    def get_command(cls, cmd_name):
        try:
            cmd = cls.commands[cmd_name]
        except KeyError:
            log_error(f"Could not remove unknown command '{cmd_name}'")
            return None
        return cmd


def initialize_commands(central_dir_path):
    # TODO: Finish!
    # TODO: Add params
    cmd_list = [
        ('add', 'select new faces', handler_add_new_embeddings, []),
        ('edit', 'edit existing faces', None, []),
        ('find', 'find individual', None, []),
        ('reclassify', 'reclassify individuals', None, []),
        ('showcluster', 'show a cluster', handler_show_cluster, []),
    ]
    for name, desc, handler, params in cmd_list:
        Command(name, desc, handler, params)


def process_command(cmd):
    # TODO: complete function(?)
    # if command in 'add':  # select new faces
    #     # TODO: complete section
    #     add_new_embeddings()
    #
    # elif command in 'edit':  # edit existing faces (or rather, existing identities)
    #     ...
    #
    # elif command in 'find':
    #     ...

    handler, params = cmd.handler, cmd.params
    handler(*params)


# ----- COMMAND PROCESSING -----

def handler_show_cluster(clusters_path):
    # TODO: Finish implementation
    should_continue = ''
    while 'n' not in should_continue:
        cluster_name, cluster_path = _user_choose_cluster(clusters_path)
        _output_cluster_content(cluster_name, cluster_path)
        ...
        should_continue = clean_str(input('Choose another cluster?\n'))


def handler_add_new_embeddings():
    """
    1. User selects new images to extract faces out of
    2. Extract faces
    3. Store embeddings in vector space


    """
    # Img Selection + Face Extraction
    face_embeddings_gen = user_choose_imgs()
    add_embeddings_to_vector_space(face_embeddings_gen)


def add_embeddings_to_vector_space(embeddings):
    """

    :param embeddings: An iterable containing the embeddings
    :return:
    """
    # TODO: Needed? What does this do, exactly??
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


def load_img_tensors_from_dir(dir_path, output_file_name=False):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param output_file_name: Whether the tensor should be yielded together with the corresponding file name
    :param dir_path: Directory containing images
    :return: Yield(!) tuples of image_names and tensors contained in this folder
    """
    # :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}. Empty = no
    # filtering
    # TODO: Needed?
    # TODO: Finish implementing
    if not output_file_name:
        for img_name in get_img_names(dir_path):
            with Image.open(dir_path + os.path.sep + img_name) as img:
                yield _to_tensor(img)
    else:
        for img_name in get_img_names(dir_path):
            with Image.open(dir_path + os.path.sep + img_name) as img:
                yield img_name, _to_tensor(img)


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


def get_img_names(dir_path):
    # TODO: Finish implementing
    # TODO: Implement recursive option?
    # TODO: Implement extension filtering?
    return list(filter(lambda name: os.path.isfile(name), os.listdir(dir_path)))


def _output_cluster_content(cluster_name, cluster_path):
    wait_for_any_input(f'Which face image in the cluster "{cluster_name}" would you like to view?')
    # TODO: finish; output faces and (-> separate function?) allow choice of image


# --- i/o helpers ---

def _user_choose_cluster(clusters_path, return_names=True):
    clusters_names_and_paths = list(get_clusters_gen(clusters_path, return_names=True))
    clusters_names = get_nth_tuple_elem(clusters_names_and_paths, n=0)
    prompt_cluster_choice(clusters_names)
    chosen_cluster_name = input()
    while chosen_cluster_name not in clusters_names:
        log_error(f'cluster "{chosen_cluster_name}" not found; Please try again.')
        prompt_cluster_choice(clusters_names)
        chosen_cluster_name = input()

    chosen_cluster_path = next(filter(lambda iterable: iterable[0] == chosen_cluster_name, clusters_names_and_paths))[1]
    if return_names:
        return chosen_cluster_name, chosen_cluster_path
    return chosen_cluster_path


def prompt_cluster_choice(clusters_names):
    temp_lim = 10
    # TODO: print clusters limited number at a time (Enter=continue)
    clusters_names = clusters_names[:temp_lim]  # TODO: remove this line
    clusters_str = '\n'.join(map(lambda string: f'- {string}', clusters_names))
    wait_for_any_input('Which cluster would you like to view? (Press any key to continue.)')
    print(clusters_str)


# ----- FILE I/O -----

def get_clusters_gen(clusters_path, return_names=True):
    file_obj_names = os.listdir(clusters_path)
    file_obj_paths = map(lambda obj_name: os.path.join(clusters_path, obj_name), file_obj_names)
    clusters_names_and_paths = filter(lambda obj_tup: os.path.isdir(obj_tup[1]), zip(file_obj_names, file_obj_paths))
    if return_names:
        return clusters_names_and_paths
    # return only paths
    return get_nth_tuple_elem(clusters_names_and_paths, n=1)

# ----- MISC -----


handlers = {
    'add': handler_add_new_embeddings,
    'showcluster': handler_show_cluster,
}

handlers_params = {
    'showcluster': [],
}
