import torch
import torchvision
from torchvision import transforms, datasets
from Logic.face_evoLVe_PyTorch.backbone.model_irse import IR_50
from facenet_pytorch import MTCNN
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm


MODEL_PATH = '../my_model/backbone_ir50_ms1m_epoch63.pth'
DATA_PATH = '../subset_cplfw/images/'
INPUT_SIZE = [112, 112]

TO_PIL_IMAGE = torchvision.transforms.ToPILImage()


class ImageFolderWithFileNames(datasets.ImageFolder):
    """Custom dataset that includes image file paths.
    Extends torchvision.datasets.ImageFolder.
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithFileNames, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        file_name = path.split('\\')[-1]
        # make a new tuple that includes the original image and its file name
        tuple_with_file_name = (original_tuple[0], file_name)
        return tuple_with_file_name


def load_data(data_dir):
    # instantiate the dataset and data loader
    # dataset = datasets.ImageFolder(root=data_dir, transform=transforms.ToTensor())
    dataset = ImageFolderWithFileNames(data_dir, transform=transforms.ToTensor())  # our custom dataset
    data_loader = torch.utils.data.DataLoader(dataset)
    return data_loader


def compute_dist_matrix(vectors):
    num_vectors = len(vectors)
    dist_matrix = torch.zeros(num_vectors, num_vectors)

    for ind1, vector1 in enumerate(vectors):
        for ind2, vector2 in enumerate(vectors):
            if ind2 <= ind1:
                continue
            cur_dist = vector1.dist(vector2)
            dist_matrix[ind1][ind2] = cur_dist
            dist_matrix[ind2][ind1] = cur_dist

    return dist_matrix


def show_dist_matrix(dist_matrix):
    plt.imshow(dist_matrix.detach().numpy(), cmap=cm.YlOrRd)
    plt.colorbar()
    plt.show()


# def load_dataset(data_path):
#     # https://stackoverflow.com/questions/50052295/how-do-you-load-images-into-pytorch-dataloader
#     train_dataset = torchvision.datasets.ImageFolder(
#         root=data_path,
#         transform=torchvision.transforms.ToTensor()
#     )
#     train_loader = torch.utils.data.DataLoader(
#         train_dataset
#     )
#     return train_loader


def save_cropped_and_aligned_images(input_size, data_path):
    mtcnn = MTCNN(image_size=input_size[0], margin=0)

    filename = "AJ_Cook/AJ_Cook_1.jpg"
    img_ = Image.open(DATA_PATH + filename)
    res_ = mtcnn(img_)


    data_loader = load_data(data_path)
    # for counter, (img_tensor, file_name_tup) in enumerate(data_loader, start=1):
    for counter, (img_tensor, img_name_tup) in enumerate(data_loader, start=1):
        if counter >= 10:
            break
        if counter % 10 == 0:
            print(counter)

        # img_tensor = img_tensor.permute(0, 2, 3, 1)
        img_name = img_name_tup[0]
        res = mtcnn(img_tensor[0])  # , save_path=f"preprocessed_faces_naive/preprocessed_{img_name}")


if __name__ == '__main__':
    save_cropped_and_aligned_images(INPUT_SIZE, DATA_PATH)
