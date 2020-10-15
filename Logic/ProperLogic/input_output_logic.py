import os
import pickle

import torch
import torchvision

from numpy import average

from PIL import Image

from facenet_pytorch import MTCNN, InceptionResnetV1

from timeit import default_timer
import logging
logging.basicConfig(level=logging.INFO)


IMAGE_PATH = os.path.join('..', 'my_test', 'facenet_Test', 'subset_cplfw_test', 'preprocessed_faces_naive')
TENSORS_PATH = 'stored_tensors'

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()
TO_TENSOR = torchvision.transforms.ToTensor()

# INPUT_SIZE = [112, 112]


# TODO: Separate extraction and loading of images!
# TODO: Learn how to time stuff well! Wrapper??

def main(imgs_dir_path, tensors_dir_path):
    # time1 = default_timer()
    # mtcnn = MTCNN(image_size=160, margin=0)
    # resnet = InceptionResnetV1(pretrained='vggface2').eval()
    # time2 = default_timer()
    # logging.info(f"Loading models: {time2 - time1}")
    #
    # time1 = default_timer()
    # imgs_loader = load_img_tensors_from_path(imgs_dir_path)
    # save_embeddings_to_path(imgs_loader, resnet, tensors_dir_path)  #TODO: (?) put mtcnn in
    # time2 = default_timer()
    # logging.info(f"Saving embeddings: {time2 - time1}")

    tensors_loader = load_embeddings_from_path(tensors_dir_path)
    dists = []
    prev_tensor = torch.zeros([1, 512])
    for tensor_path, tensor in tensors_loader:
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


def load_img_tensors_from_path(dir_path, img_extensions=None):
    """
    Yield all images in the given directory.
    If img_img_extensions is empty, all files are assumed to be images. Otherwise, only files with extensions appearing
    in the set will be returned.

    :param dir_path: Directory containing images
    :param img_extensions: Set of lower-case file extensions considered images, e.g. {'jpg', 'png', 'gif'}
    :return: Yield(!) tuples of image_names and tensors contained in this folder
    """
    # TODO: Implement(?) img_extensions functionality
    if img_extensions is None:
        img_extensions = set()
    img_names, num_imgs = get_and_count_img_names(dir_path)

    for counter, img_name in enumerate(img_names, start=1):
        with Image.open(os.path.join(dir_path, img_name)) as img:
            yield img_name, img  # TO_TENSOR(img).unsqueeze(0)


def save_embeddings_to_path(imgs_loader, face_embedder, save_path, face_extractor=None):
    """
    Save face embeddings of images in the image loader.

    :param face_embedder:
    :param face_extractor:
    :param save_path:
    :param imgs_loader:
    :return:
    """

    # img_old_clooney = Image.open("age_imgs/old-george-clooney.jpg")
    #
    # # Get cropped and pre-whitened image tensor
    # img_old_cropped = mtcnn(img_old_clooney, save_path="age_testrun_imgs/img_old_cropped.jpg")  # mtcnn
    #
    # # Calculate embedding (unsqueeze to add batch dimension)
    # img_old_embedding = resnet(img_old_cropped.unsqueeze(0))

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    elif not os.path.isdir(save_path):
        raise OSError(f'object named {save_path} already exists, directory of same name cannot be created')

    def _extract_face_and_save_embedding(img_name, img, should_extract_face=True):
        # Get pre-whitened and cropped image tensor of the face
        face_tensor = face_extractor(img) if should_extract_face else TO_TENSOR(img)
        # Compute face embedding
        face_embedding = face_embedder(face_tensor.unsqueeze(0))

        tensor_save_path = os.path.join(save_path, f"embedding_{img_name.split('.')[0]}.pt")
        torch.save(face_embedding, tensor_save_path, pickle_protocol=pickle.DEFAULT_PROTOCOL)

    # TODO: PROPERLY Vectorize this loop(?)
    should_extract_face = face_extractor is not None
    for counter, (img_name, img) in enumerate(imgs_loader, start=1):
        if counter % 50 == 0:
            logging.info(f"loop number: {counter}")
        _extract_face_and_save_embedding(img_name, img, should_extract_face)


def load_embeddings_from_path(tensors_path, tensor_extensions=None):
    """
    Load face embeddings (tensors) from given path.

    :param :
    :return:
    """
    # TODO: Implement functionality to choose which tensors to load
    if tensor_extensions is None:
        tensor_extensions = set()

    file_names = filter(lambda obj_name: os.path.isfile(os.path.join(tensors_path, obj_name)),
                        os.listdir(tensors_path))
    if tensor_extensions:
        file_names = filter(lambda obj_name: get_file_extension(obj_name) in tensor_extensions, file_names)

    file_paths = map(lambda file_name: os.path.join(tensors_path, file_name), file_names)
    for counter, tensor_path in enumerate(file_paths, start=1):
        yield tensor_path, torch.load(tensor_path)


def get_file_extension(file_name):
    return file_name.split(os.path.extsep)[-1]


def strip_file_extension(file_name):
    return os.path.extsep.join(file_name.split(os.path.extsep)[:-1])


def append_file_extension(file_name, extension):
    return file_name + os.path.extsep + extension


if __name__ == '__main__':
    main(IMAGE_PATH, TENSORS_PATH)
    # tensors = load_img_tensors_from_path(TENSORS_PATH)
    # tensor = list(tensors)[0][1]
    # ...
