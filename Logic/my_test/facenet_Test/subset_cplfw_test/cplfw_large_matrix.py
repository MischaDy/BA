import os

import torch
import torchvision
from Logic.face_evoLVe_PyTorch.backbone.model_irse import IR_50
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import logging
from Logic.my_test.face_evoLVe_Test.subset_cplfw_test.get_size_of_obj import total_size

logging.basicConfig(level=logging.INFO)


MODEL_PATH = '../my_model/backbone_ir50_ms1m_epoch63.pth'
DATA_PATH = 'preprocessed_faces_naive/'

# TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()

INPUT_SIZE = [112, 112]


def main(root, model_path, input_size):
    log("main", is_starting=True)

    img_names, num_imgs = get_and_count_img_names(root)
    tensors_loader = load_img_tensors_from_dir(root, img_names)

    vectors = compute_face_embeddings(tensors_loader, model_path, input_size)
    dist_matrix = compute_dist_matrix(vectors, num_imgs)
    show_dist_matrix(dist_matrix)

    log("main", is_starting=False)


def get_and_count_img_names(dir_path, img_extensions=set()):
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
    log("load_img_tensors_from_dir", is_starting=True)
    for i, img_name in enumerate(img_names, start=1):
        if i % 10 == 0:
            log_counter("load_img_tensors_from_dir", i)
        with Image.open(f"{dir_path}/{img_name}") as img:
            yield img_name, TO_TENSOR(img).unsqueeze(0)
    log("load_img_tensors_from_dir", is_starting=False)


def compute_face_embeddings(tensors_loader, model_path, input_size):
    """
    Yield face embeddings of images in the image loader.

    :param tensors_loader:
    :param model_path:
    :param input_size:
    :return:
    """
    log("compute_face_embeddings", is_starting=True)
    device = torch.device('cpu')
    model = IR_50(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))  # 'cpu'
    model.eval()

    for counter, (img_name, tensor) in enumerate(tensors_loader, start=1):
        if counter % 10 == 0: # or counter >= 110:
            log_counter("compute_face_embeddings", counter)
        yield model(tensor)
    log("compute_face_embeddings", is_starting=False)


def compute_dist_matrix(vectors, num_vectors):
    log("compute_dist_matrix", is_starting=True)
    dist_matrix = torch.zeros(num_vectors, num_vectors)

    counter = 1
    for ind1, vector1 in enumerate(vectors):
        for ind2, vector2 in enumerate(vectors):
            if counter % 10 == 0:
                log_counter("compute_dist_matrix", counter)
            if ind2 <= ind1:
                continue
            cur_dist = vector1.dist(vector2)
            dist_matrix[ind1][ind2] = cur_dist
            dist_matrix[ind2][ind1] = cur_dist

            counter += 1
    log("compute_dist_matrix", is_starting=False)

    return dist_matrix


def show_dist_matrix(dist_matrix):
    log("show_dist_matrix", is_starting=True)
    plt.imshow(dist_matrix.detach().numpy(), cmap=cm.YlOrRd)
    plt.colorbar()
    plt.show()
    log("show_dist_matrix", is_starting=False)


def log(func_name, is_starting):
    logging.info(f"------------ {'START' if is_starting else 'END'}: {func_name} ------------")


def log_counter(func_name, log_counter):
    logging.info(f"{func_name} -- {log_counter}")


if __name__ == '__main__':
    main(DATA_PATH, MODEL_PATH, INPUT_SIZE)
