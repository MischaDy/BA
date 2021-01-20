from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


def main():
    print(f"{deti() - _start_time}:  ------- FUNCTION START -------\n")
    print(f"{deti() - _start_time}:  ------- Initialization -------\n")

    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    mtcnn = MTCNN(image_size=160, margin=0, keep_all=True)

    difficulties = ["easy", "hard"]
    classes = ["babies", "group"]
    orig_path = "group_imgs/"
    file_ending = ".jpg"

    # Open Files
    print(f"{deti() - _start_time}:  ------- Opening Files -------\n")
    file_names = [f"{class_}-{diff}{file_ending}"
                  for class_ in classes
                  for diff in difficulties]
    imgs = [Image.open(orig_path + file_name) for file_name in file_names]

    # Get cropped and pre-whitened image tensors
    print(f"{deti() - _start_time}:  ------- Finding and cropping Faces -------\n")
    base_save_path = "group_testrun_imgs/"
    imgs_cropped = [mtcnn(img, save_path=build_save_path(base_save_path, file_name))
                    for img, file_name in zip(imgs, file_names)]

    # #  Calculate embedding (unsqueeze to add batch dimension)
    # imgs_embeddings = [resnet(img_cropped.unsqueeze(0))
    #                    for img_cropped in imgs_cropped]

    # Or, if using for VGGFace2 classification
    print(f"{deti() - _start_time}:  ------- Classifying -------\n")
    resnet.classify = True
    imgs_probs = {}
    for file_name, imgs in zip(file_names, imgs_cropped):
        imgs_probs[file_name] = [resnet(img.unsqueeze(0)) for img in imgs]

    print(f"{deti() - _start_time}:  ------- FUNCTION DONE -------")

    # # print probabilities
    # for file_name in file_names:
    #     print(f"{file_name}:  {imgs_probs[file_name]}\n")

    return imgs_probs


def build_save_path(base_save_path, file_name):
    subdir_name = ".".join(file_name.split(".")[:-1]) + "/"
    return f"{base_save_path}{subdir_name}cropped-{file_name}"


if __name__ == "__main__":
    from timeit import default_timer as deti
    _start_time = deti()
    print(f"{deti() - _start_time}:  ------- PROGRAM START -------\n")
    main()
    print(f"{deti() - _start_time}:  ------- PROGRAM END -------\n")
