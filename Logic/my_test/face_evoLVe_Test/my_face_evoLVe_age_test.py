from Logic.face_evoLVe_PyTorch.align.detector import detect_faces
from Logic.face_evoLVe_PyTorch.align.visualization_utils import show_results
from PIL import Image

print("START\n")

main_img_path = "age_imgs/"
save_img_path = "age_testrun_imgs/"
file_names = ["old-george-clooney.jpg",
              "younger-george-clooney.jpg",
              "lookalike-george-clooney.jpg"]

for file_name in file_names:
      img = Image.open(main_img_path + file_name)
      bounding_boxes, landmarks = detect_faces(img)
      result_img = show_results(img, bounding_boxes, landmarks)
      result_img.show()
      result_img.save(f"{save_img_path}detected-{file_name}")

print("\nEND")
