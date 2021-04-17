from Logic.face_evoLVe_PyTorch.align.detector import detect_faces
from Logic.face_evoLVe_PyTorch.align.visualization_utils import show_results
from PIL import Image


from facenet_pytorch import MTCNN
mtcnn = MTCNN(image_size=160, margin=0)

# main_img_path = "age_imgs/"
# save_img_path = "age_testrun_imgs/"
# file_names = ["old-george-clooney.jpg",
#               "younger-george-clooney.jpg",
#               "lookalike-george-clooney.jpg"]
#
# for file_name in file_names:
#       img = Image.open(main_img_path + file_name)
#       bounding_boxes, landmarks = detect_faces(img)
#       result_img = show_results(img, bounding_boxes, landmarks)
#       result_img.show()
#       result_img.save(f"{save_img_path}detected-{file_name}")


def main():
    print(f"{deti() - _start_time}:  ------- FUNCTION START -------\n")
    print(f"{deti() - _start_time}:  ------- Initialization -------\n")

    # mtcnn = MTCNN(image_size=160, margin=0, keep_all=True)

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

    base_save_path = "group_testrun_imgs/"
    # Get detected and pre-whitened image tensors
    print(f"{deti() - _start_time}:  ------- Finding and cropping Faces -------\n")
    properties_detected = []
    # for img, file_name in zip(imgs, file_names):
    #     print(f"------- ------- Currently:  {file_name}")
    #     properties_detected.append(detect_faces(img))
    img, file_name = (imgs[2], file_names[2])
    print(f"------- ------- Currently:  {file_name}")
    properties_detected.append(mtcnn.detect(img, True))
    # properties_detected.append(detect_faces(img))

    print(f"{deti() - _start_time}:  ------- Drawing result images -------\n")
    result_imgs = []
    for ind, (img, property_detected) in enumerate(zip(imgs, properties_detected)):
        print(f"------- ------- Currently:  {ind}")
        result_imgs.append(show_results(img, *property_detected))

    print(f"{deti() - _start_time}:  ------- Saving result images -------\n")
    for result_img, file_name in zip(result_imgs, file_names):
        print(f"------- ------- Currently:  {file_name}")
        result_img.save(build_save_path(base_save_path, file_name))
        result_img.show()

    # #  Calculate embedding (unsqueeze to add batch dimension)
    # imgs_embeddings = [resnet(img_detected.unsqueeze(0))
    #                    for img_detected in imgs_detected]

    # # Or, if using for VGGFace2 classification
    # print(f"{deti() - _start_time}:  ------- Classifying -------\n")
    # resnet.classify = True
    # imgs_probs = {}
    # for file_name, imgs in zip(file_names, imgs_detected):
    #     imgs_probs[file_name] = [resnet(img.unsqueeze(0)) for img in imgs]
    #
    # print(f"{deti() - _start_time}:  ------- FUNCTION DONE -------")

    #
    # return imgs_probs


def build_save_path(base_save_path, file_name):
    subdir_name = ".".join(file_name.split(".")[:-1]) + "/"
    return f"{base_save_path}{subdir_name}detected-{file_name}"


if __name__ == "__main__":
    from timeit import default_timer as deti
    _start_time = deti()
    print(f"{deti() - _start_time}:  ------- PROGRAM START -------\n")
    main()
    print(f"{deti() - _start_time}:  ------- PROGRAM END -------\n")
