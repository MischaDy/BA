"""
Program containing the main application logic.
"""
import os

import torchvision
from PIL import Image


EMBEDDING_STORAGE_PATH

IMG_PATH = "Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive"
TO_TENSOR = torchvision.transforms.ToTensor()

TERMINATING_TERMS = ('halt', 'stop', 'quit', 'exit', )
COMMANDS = {'select New Faces': 'nf', 'edit Existing Faces': 'ef', }




# TODO: Test saving and loading tensors, check size!
# TODO: Consistent naming
# TODO: What should / shouldn't be private?
# TODO: Make Commands to Enum
# TODO: Add comments & docstrings


def main():
    command = ''
    while command not in TERMINATING_TERMS:
        command = get_user_command()
        if command == 'nf':  # select new faces
            # TODO: complete
            add_new_embeddings()

        elif command == 'ef':  # edit existing faces
            ...

        else:
            raise NotImplementedError(f'known, but not yet implemented command {command}')


def get_user_command():
    command = _get_user_command_subfunc()
    while command not in COMMANDS.values():
        command = _get_user_command_subfunc()
    return command


def _get_user_command_subfunc():
    print('\nWhat would you like to do next?')
    print_command_options()
    return input()


def print_command_options():
    print('\n'.join(f"To {command}, type '{abbreviation}'." for command, abbreviation in COMMANDS))


def add_new_embeddings():
    """
    1. User selects new images to extract faces out of
    2. Faces are extracted


    """
    # Img Selection + Face Extraction
    face_embeddings_gen = user_choose_imgs()
    add_to_embedding_space(face_embeddings_gen)


def add_to_embedding_space(embeddings):
    """

    :param embeddings: An iterable containing the embeddings
    :return:
    """



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


if __name__ == "__main__":
    main()
