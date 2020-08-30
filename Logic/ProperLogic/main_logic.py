"""
Program containing the main application logic.
"""
import os

import torchvision
from PIL import Image


IMG_PATH = "Logic/my_test/facenet_Test/subset_cplfw_test/preprocessed_faces_naive"
TO_TENSOR = torchvision.transforms.ToTensor()


# TODO: Test saving and loading tensors, check size!
# TODO: Consistent naming

def main():
    """
    1. User selects new images to extract faces out of.
    2. Faces are extracted


    """
    face_embeddings = user_choose_imgs()
    print()







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


def load_img_tensors_from_dir(dir_path, img_extensions=set()):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param dir_path: Directory containing images
    :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}. Empty = no
    filtering
    :return: Yield(!) tuples of image_names and tensors contained in this folder
    """
    # TODO: Finish implementing
    for i, img_name in enumerate(get_img_names(dir_path, img_extensions), start=1):
        with Image.open(dir_path + os.path.sep + img_name) as img:
            yield img_name, _to_tensor(img)


def _to_tensor(img):
    return TO_TENSOR(img).unsqueeze(0)


def get_img_names(dir_path, recursive=False, img_extensions=set()):
    # TODO: Finish implementing
    # TODO: Implement recursive option
    # TODO: Implement extension filtering
    return list(filter(lambda name: os.path.isfile(name), os.listdir(dir_path)))





if __name__ == "__main__":
    main()
