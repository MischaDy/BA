from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image

import torch


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


def print_dist_matrix(dist_matrix, vector_names):
    print(vector_names)
    print(dist_matrix)


resnet = InceptionResnetV1(pretrained='vggface2').eval()
mtcnn = MTCNN(image_size=160, margin=0)


img_old_clooney = Image.open("age_imgs/old-george-clooney.jpg")
img_younger_clooney = Image.open("age_imgs/younger-george-clooney.jpg")
img_lookalike_clooney = Image.open("age_imgs/lookalike-george-clooney.jpg")

# Get cropped and pre-whitened image tensor
img_old_cropped = mtcnn(img_old_clooney, save_path="age_testrun_imgs/img_old_cropped.jpg")  # mtcnn
img_younger_cropped = mtcnn(img_younger_clooney, save_path="age_testrun_imgs/img_younger_cropped.jpg")
img_lookalike_cropped = mtcnn(img_lookalike_clooney, save_path="age_testrun_imgs/img_lookalike_cropped.jpg")

# Calculate embedding (unsqueeze to add batch dimension)
img_old_embedding = resnet(img_old_cropped.unsqueeze(0))
img_younger_embedding = resnet(img_younger_cropped.unsqueeze(0))
img_lookalike_embedding = resnet(img_lookalike_cropped.unsqueeze(0))

# Or, if using for VGGFace2 classification
resnet.classify = True
img_old_probs = resnet(img_old_cropped.unsqueeze(0))
img_younger_probs = resnet(img_younger_cropped.unsqueeze(0))
img_lookalike_probs = resnet(img_lookalike_cropped.unsqueeze(0))


probs_list = (img_old_probs, img_younger_probs, img_lookalike_probs)
probs_dist_matrix = compute_dist_matrix(probs_list)
vector_names = ("Old Clooney", "Younger Clooney", "Lookalike")

print_dist_matrix(probs_dist_matrix, vector_names)

embeddings_list = (img_old_embedding, img_younger_embedding, img_lookalike_embedding)
embeddings_dist_matrix = compute_dist_matrix(embeddings_list)
vector_names = ("Old Clooney", "Younger Clooney", "Lookalike")

print_dist_matrix(embeddings_dist_matrix, vector_names)





# print(f"Old Clooney:  {img_old_probs}\n"
#       + f"Younger Clooney:  {img_younger_probs}\n"
#       + f"Lookalike Clooney:  {img_lookalike_probs}")


# mtcnn = MTCNN(image_size=160, margin=42)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()




