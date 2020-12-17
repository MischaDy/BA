import torch
from torchvision import transforms
from Logic.face_evoLVe_PyTorch.backbone.model_irse import IR_50
from facenet_pytorch import MTCNN
from PIL import Image

import matplotlib.pyplot as plt
import matplotlib.cm as cm


MODEL_PATH = '../my_model/backbone_ir50_ms1m_epoch63.pth'
PIL_TO_TENSOR = transforms.ToTensor()


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


input_size = [112, 112]

mtcnn = MTCNN(image_size=input_size[0], margin=0)

device = torch.device('cpu')
model = IR_50(input_size)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()


# img_old_clooney = Image.open("../age_imgs/old-george-clooney.jpg")
# img_younger_clooney = Image.open("../age_imgs/younger-george-clooney.jpg")
# img_lookalike_clooney = Image.open("../age_imgs/lookalike-george-clooney.jpg")

# # Get cropped and pre-whitened image tensor
# img_old_cropped = mtcnn(img_old_clooney, save_path="../age_testrun_imgs/img_old_cropped.jpg")  # mtcnn
# img_younger_cropped = mtcnn(img_younger_clooney, save_path="../age_testrun_imgs/img_younger_cropped.jpg")
# img_lookalike_cropped = mtcnn(img_lookalike_clooney, save_path="../age_testrun_imgs/img_lookalike_cropped.jpg")

# # Get cropped and pre-whitened image tensor
img_old_cropped = PIL_TO_TENSOR(Image.open("../age_testrun_imgs/img_old_cropped.jpg"))
img_younger_cropped = PIL_TO_TENSOR(Image.open("../age_testrun_imgs/img_younger_cropped.jpg"))
img_lookalike_cropped = PIL_TO_TENSOR(Image.open("../age_testrun_imgs/img_lookalike_cropped.jpg"))


# Calculate embedding (unsqueeze to add batch dimension)
img_old_embedding = model(img_old_cropped.unsqueeze(0))
img_younger_embedding = model(img_younger_cropped.unsqueeze(0))
img_lookalike_embedding = model(img_lookalike_cropped.unsqueeze(0))


embeddings_list = (img_old_embedding, img_younger_embedding, img_lookalike_embedding)
embeddings_dist_matrix = compute_dist_matrix(embeddings_list)
# vector_names = ("Old Clooney", "Younger Clooney", "Lookalike")
show_dist_matrix(embeddings_dist_matrix)  # , vector_names)
