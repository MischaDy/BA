from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


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

print(f"Old Clooney:  {img_old_probs}\n"
      f"Younger Clooney:  {img_younger_probs}\n"
      f"Lookalike Clooney:  {img_lookalike_probs}")

print(img_lookalike_probs[0][0])


# mtcnn = MTCNN(image_size=160, margin=42)
# resnet = InceptionResnetV1(pretrained='vggface2').eval()
