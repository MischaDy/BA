import os
import pickle

import torch
import torchvision

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from facenet_pytorch import MTCNN, InceptionResnetV1


DATA_PATH = os.path.join('subset_cplfw_test', 'preprocessed_faces_naive')
SAVE_DIR = os.path.join('subset_cplfw_test', 'saved_tensors')

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()

INPUT_SIZE = [112, 112]


resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=0)


def main(root, save_dir):
    img_names, num_imgs = get_and_count_img_names(root)
    tensors_loader = load_img_tensors_from_dir(root, img_names)
    save_img_tensors_to_dir(tensors_loader, save_dir)


def get_and_count_img_names(dir_path, img_extensions=None):
    if img_extensions is None:
        img_extensions = set()

    dir_path = dir_path.rstrip('/')
    if len(img_extensions) == 0:
        img_names = (elem for elem in os.listdir(dir_path)
                     if os.path.isfile(f"{dir_path}/{elem}"))
        num_images = len(list(elem for elem in os.listdir(dir_path)
                              if os.path.isfile(f"{dir_path}/{elem}")))
    else:
        img_names = (elem for elem in os.listdir(dir_path)
                     if os.path.isfile(f"{dir_path}/{elem}")
                     and os.path.splitext(elem)[-1].lower().lstrip('.') in img_extensions)
        num_images = len(list(elem for elem in os.listdir(dir_path)
                              if os.path.isfile(f"{dir_path}/{elem}")
                              and os.path.splitext(elem)[-1].lower().lstrip('.') in img_extensions))
    return img_names, num_images


def load_img_tensors_from_dir(dir_path, img_names, img_extensions=set()):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param dir_path: Directory containing images
    :param img_names: Names of the images in dir_path
    :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}
    :return: Yield(!) tuples of image_names and tensors contained in this folder
    """
    for i, img_name in enumerate(img_names, start=1):
        with Image.open(f"{dir_path}/{img_name}") as img:
            yield img_name, TO_TENSOR(img).unsqueeze(0)


def save_img_tensors_to_dir(tensors_loader, save_dir):
    """
    Save face embeddings of images in the image loader.

    :param save_dir:
    :param tensors_loader:
    :return:
    """

    # img_old_clooney = Image.open("age_imgs/old-george-clooney.jpg")
    #
    # # Get cropped and pre-whitened image tensor
    # img_old_cropped = mtcnn(img_old_clooney, save_path="age_testrun_imgs/img_old_cropped.jpg")  # mtcnn
    #
    # # Calculate embedding (unsqueeze to add batch dimension)
    # img_old_embedding = resnet(img_old_cropped.unsqueeze(0))
    #
    # embeddings_list = (img_old_embedding)
    # embeddings_dist_matrix = compute_dist_matrix(embeddings_list)
    # vector_names = ("Old Clooney", "Younger Clooney", "Lookalike")

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    elif not os.path.isdir(save_dir):
        raise OSError(f'object named {save_dir} already exists, directory of same name cannot be created')

    for counter, (img_name, tensor) in enumerate(tensors_loader, start=1):
        tensor_save_path = os.path.join(save_dir, f"tensor_{img_name.split('.')[0]}.pt")
        torch.save(tensor, tensor_save_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)
        # break
        # mtcnn(tensor, save_path=f"{}.jpg")


if __name__ == '__main__':
    main(DATA_PATH, SAVE_DIR)
