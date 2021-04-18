from facenet_pytorch.models.inception_resnet_v1 import InceptionResnetV1
from Logic.ProperLogic.models_modules.altered_mtcnn import AlteredMTCNN


class Models:
    altered_mtcnn = AlteredMTCNN(image_size=160, margin=0, post_process=False, select_largest=False, keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
