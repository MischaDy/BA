import os

# import torch
import torchvision
# from torchvision import transforms, datasets
# from Logic.face_evoLVe_PyTorch.backbone.model_irse import IR_50
from facenet_pytorch import MTCNN
from PIL import Image

# import matplotlib.pyplot as plt
# import matplotlib.cm as cm


MODEL_PATH = '../my_model/backbone_ir50_ms1m_epoch63.pth'
DATA_PATH = '../subset_cplfw/images/'
INPUT_SIZE = [112, 112]

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()


# def compute_dist_matrix(vectors):
#     num_vectors = len(vectors)
#     dist_matrix = torch.zeros(num_vectors, num_vectors)
#
#     for ind1, vector1 in enumerate(vectors):
#         for ind2, vector2 in enumerate(vectors):
#             if ind2 <= ind1:
#                 continue
#             cur_dist = vector1.dist(vector2)
#             dist_matrix[ind1][ind2] = cur_dist
#             dist_matrix[ind2][ind1] = cur_dist
#
#     return dist_matrix
#
#
# def show_dist_matrix(dist_matrix):
#     plt.imshow(dist_matrix.detach().numpy(), cmap=cm.YlOrRd)
#     plt.colorbar()
#     plt.show()


def load_images(root):
    """
    Assumes directory structure like this (file names are irrelevant):
    root
    |-- subdir_1
    |   |-- img_1a
    |   |-- img_1b
    :   :
    |
    |-- subdir_n
        |-- img_na
        |-- img_nb
        :

    :param root: Root directory containing the subfolders
    :return: Generator returning generators iterating over images of each subfolder
    """
    root = root.rstrip('/')
    subdirs = (elem for elem in os.listdir(root) if os.path.isdir(f"{root}/{elem}"))

    for subdir in subdirs:
        for (img_name, img) in load_images_from_dir(f"{root}/{subdir}"):
            yield img_name, img


def load_images_from_dir(dir_path, img_extensions=set()):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}
    :param dir_path: Directory containing images
    :return: Yield(!) tuples of image_names and PIL images contained in this folder
    """
    dir_path = dir_path.rstrip('/')
    if len(img_extensions) == 0:
        img_names = (elem for elem in os.listdir(dir_path)
                     if os.path.isfile(f"{dir_path}/{elem}"))
    else:
        img_names = (elem for elem in os.listdir(dir_path)
                     if os.path.isfile(f"{dir_path}/{elem}")
                     and get_file_extension(elem) in img_extensions)

    for img_name in img_names:
        with Image.open(f"{dir_path}/{img_name}") as img:
            yield img_name, img


def get_file_extension(file_name):
    return os.path.splitext(file_name)[-1].lower().lstrip('.')


def save_cropped_and_aligned_images(input_size, data_path):
    mtcnn = MTCNN(image_size=input_size[0], margin=0)

    # for counter, (img_tensor, file_name_tup) in enumerate(data_loader, start=1):
    data_loader = load_images(data_path)
    for counter, (img_name, img) in enumerate(data_loader, start=1):
        # if counter >= 10:
        #     break
        if counter % 50 == 0:
            print(counter)

        mtcnn(img, save_path=f"preprocessed_faces_naive/preprocessed_{img_name}")


# img_old_clooney = Image.open("../subset_cplfw/")
# # Get cropped and pre-whitened image tensor
# img_old_cropped = mtcnn(img_old_clooney, save_path="../age_testrun_imgs/img_old_cropped.jpg")  # mtcnn


if __name__ == '__main__':
    save_cropped_and_aligned_images(INPUT_SIZE, DATA_PATH)
