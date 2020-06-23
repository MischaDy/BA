import os
import psutil

import torch
import torchvision

from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from facenet_pytorch import MTCNN, InceptionResnetV1

from multiprocess import Process, Pool, Lock, Manager, cpu_count, freeze_support

import logging
logging.basicConfig(level=logging.INFO)


DATA_PATH = 'subset_cplfw_test/preprocessed_faces_naive/'

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()

INPUT_SIZE = [112, 112]


def limit_cpu(priority=psutil.BELOW_NORMAL_PRIORITY_CLASS):
    # https://stackoverflow.com/questions/42103367/limit-total-cpu-usage-in-python-multiprocessing
    p = psutil.Process(os.getpid())
    # set to lowest priority, this is windows only, on Unix use ps.nice(19)
    p.nice(priority)


def main(root, input_size):
    img_names, num_imgs = get_and_count_img_names(root)
    tensors_loader = load_img_tensors_from_dir(root, img_names)

    vectors = compute_face_embeddings(tensors_loader)
    dist_matrix = compute_dist_matrix(vectors, num_imgs)
    show_dist_matrix(dist_matrix)


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
    for i, img_name in enumerate(img_names, start=1):
        with Image.open(f"{dir_path}/{img_name}") as img:
            yield img_name, TO_TENSOR(img).unsqueeze(0)


def compute_face_embeddings(tensors_loader):
    """
    Yield face embeddings of images in the image loader.

    :param tensors_loader:
    :return:
    """

    # iterate over cropped and pre-whitened image tensor
    for counter, (img_name, tensor) in enumerate(tensors_loader, start=1):
        yield resnet(tensor)  # .unsqueeze(0))

    """
    Not working:
    - tensor.unsqueeze(0)
    
    """

    # # Or, if using for VGGFace2 classification
    # resnet.classify = True
    # img_old_probs = resnet(img_old_cropped.unsqueeze(0))
    # img_younger_probs = resnet(img_younger_cropped.unsqueeze(0))
    # img_lookalike_probs = resnet(img_lookalike_cropped.unsqueeze(0))
    #
    # probs_list = (img_old_probs, img_younger_probs, img_lookalike_probs)
    # probs_dist_matrix = compute_dist_matrix(probs_list)
    # vector_names = ("Old Clooney", "Younger Clooney", "Lookalike")


def compute_dist_matrix(vectors, num_vectors):
    # limit_cpu()
    pass







def compute_dist_matrix_part(first_loop_values, second_loop_values, compute_dist):
    # TODO: Rename!
    dist_matrix_part = torch.zeros(len(first_loop_values), len(second_loop_values))

    for ind1, val1 in enumerate(first_loop_values):
        for ind2, val2 in enumerate(second_loop_values):
            # if ind2 % 50 == 1:
            #     lock = queue.get(block=True)
            #     lock.acquire()
            #     logging.info(f"{ind1}, {ind2}")
            #     lock.release()
            #     queue.put(lock)
            dist_matrix_part[ind1][ind2] = compute_dist(val1, val2)
    return dist_matrix_part


def compute_dist_matrix_part2(vectors_part, num_vectors):
    """

    :param vectors_part:
    :param num_vectors: Number of vectors in generator vectors_part
    :return: Distance matrix of these vector parts
    """
    dist_matrix_part = torch.zeros(num_vectors, num_vectors)
    for ind1, vector1 in enumerate(vectors_part):
        for ind2, vector2 in enumerate(vectors_part):
            # matrix symmetrical, vector needn't compare to itself or previous ones
            if ind2 <= ind1:
                continue
            if ind2 % 50 == 1:
                logging.info(f"{ind1}, {ind2}")
            cur_dist = vector1.dist(vector2)
            dist_matrix_part[ind1][ind2] = cur_dist
            dist_matrix_part[ind2][ind1] = cur_dist

    return dist_matrix_part


def show_dist_matrix(dist_matrix):
    plt.imshow(dist_matrix.detach().numpy(), cmap=cm.YlOrRd)
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(image_size=160, margin=0)

    main(DATA_PATH, INPUT_SIZE)
