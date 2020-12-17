# https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d
import torch
from torchvision import datasets, transforms


class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths.
    Extends torchvision.datasets.ImageFolder.
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes image of original and the path
        tuple_with_path = (original_tuple[0], path)
        return tuple_with_path


def load_data(data_dir):
    # instantiate the dataset and dataloader
    dataset = ImageFolderWithPaths(data_dir, transform=transforms.ToTensor())  # our custom dataset
    dataloader = torch.utils.data.DataLoader(dataset)
    return dataloader


if __name__ == '__main__':
    data_dir = "your/data_dir/here"
    dataloader = load_data(data_dir)

    # iterate over data
    for inputs, labels, paths in dataloader:
        # use the above variables freely
        print(inputs, labels, paths)
