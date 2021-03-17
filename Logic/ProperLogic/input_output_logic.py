import os
import pickle

import torch
import torchvision

from numpy import average

from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

import logging

from Logic.ProperLogic.cluster import Cluster

logging.basicConfig(level=logging.INFO)


IMAGE_PATH = os.path.join('..', 'my_test', 'facenet_Test', 'subset_cplfw_test', 'preprocessed_faces_naive')
embeddings_PATH = 'stored_embeddings'

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()


# TODO: How to fuse with the DBs?
#       --> DB handles DB-internals, while this module use DB module to load and store stuff for other modules?
# TODO: Remember to CREATE directories if they don't already exist! (Don't assume existence)
# TODO: Separate extraction and loading of images!
# TODO: (Learn how to time stuff well! Wrapper?)


def main(imgs_dir_path, embeddings_dir_path):
    mtcnn = MTCNN(image_size=160, margin=0)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()

    _test_saving_embeddings(imgs_dir_path, embeddings_dir_path, resnet)
    _test_loading_embeddings(embeddings_dir_path)


def load_clusters_from_db(db_manager):
    cluster_parts = db_manager.get_cluster_parts()
    clusters = [Cluster(embeddings=[embedding], embeddings_ids=[embedding_id], cluster_id=cluster_id, label=label,
                        center_point=center_point)
                for cluster_id, label, center_point, embedding, embedding_id in cluster_parts]
    return clusters


def _test_saving_embeddings(imgs_dir_path, embeddings_dir_path, resnet):
    imgs_loader = load_imgs_from_path(imgs_dir_path)
    save_embeddings_to_path(imgs_loader, resnet, embeddings_dir_path)  # TODO: (?) put mtcnn in


def _test_loading_embeddings(embeddings_dir_path):
    embeddings_loader = ...  # _load_embeddings_from_path(embeddings_dir_path, yield_paths=True)
    dists = []
    prev_tensor = torch.zeros([1, 512])
    for tensor_path, tensor in embeddings_loader:
        dists.append(float(torch.dist(prev_tensor, tensor)))
        prev_tensor = tensor

    print(average(dists))


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


def load_imgs_from_path(dir_path):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param dir_path: Directory containing images
    :return: Yield(!) tuples of image_names and embeddings contained in this folder
    """
    # :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}
    img_names, num_imgs = get_and_count_img_names(dir_path)

    for counter, img_name in enumerate(img_names, start=1):
        with Image.open(os.path.join(dir_path, img_name)) as img:
            yield img_name, img  # TO_TENSOR(img).unsqueeze(0)


def save_embeddings_to_path(imgs_loader, face_embedder, save_path, face_extractor=None):
    """
    Extract and save face embeddings of images in the image loader. If face_extractor is None, then the extraction step
    is skipped.

    :param face_embedder:
    :param face_extractor:
    :param save_path:
    :param imgs_loader:
    :return:
    """

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    elif not os.path.isdir(save_path):
        raise OSError(f'non-directory object named {save_path} already exists, '
                      'directory of same name cannot be created')

    def _extract_face_and_save_embedding(img_name, img, should_extract_face=True):
        # Get pre-whitened and cropped image tensor of the face
        face_tensor = face_extractor(img) if should_extract_face else TO_TENSOR(img)
        # Compute face embedding
        face_embedding = face_embedder(face_tensor.unsqueeze(0))

        tensor_save_path = os.path.join(save_path, f"embedding_{img_name.split('.')[0]}.pt")
        torch.save(face_embedding, tensor_save_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)

    # TODO: PROPERLY vectorize this loop(?)
    should_extract_face = face_extractor is not None
    for counter, (img_name, img) in enumerate(imgs_loader, start=1):
        if counter % 50 == 0:
            logging.info(f"extract & save loop number: {counter}")
        _extract_face_and_save_embedding(img_name, img, should_extract_face)


if __name__ == '__main__':
    main(IMAGE_PATH, embeddings_PATH)
