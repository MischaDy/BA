from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from facenet_pytorch.models.mtcnn import MTCNN


class Models:
    mtcnn = MTCNN(image_size=160, margin=0, select_largest=False, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
