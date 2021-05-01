from PIL import Image
from Logic.xxxface_evoLVe_PyTorch.align import detector, visualization_utils


detect_faces = detector.detect_faces
show_results = visualization_utils.show_results


img = Image.open('group-hard.jpg')  # modify the image path to yours
bounding_boxes, landmarks = detect_faces(img)  # detect bboxes and landmarks for all faces in the image
print("res:")
ic = show_results(img, bounding_boxes, landmarks)  # visualize the results
ic.show()
print("^res")
