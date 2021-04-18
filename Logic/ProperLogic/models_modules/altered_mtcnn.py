import torch
from facenet_pytorch.models.mtcnn import MTCNN, fixed_image_standardization
from facenet_pytorch.models.utils.detect_face import get_size, crop_resize


class AlteredMTCNN(MTCNN):
    def forward_return_results(self, img):
        """Run MTCNN face detection on a PIL image or numpy array. This method performs both
        detection and extraction of faces, returning PIL images representing detected faces rather
        than the bounding boxes. To access bounding boxes, see the MTCNN.detect() method.

        This method is nearly identical to the forward method provided by the MTCNN class, but with
        slightly fewer capabilities and returning the detected images rather than (possibly) saving
        them somewhere.

        Arguments:
            img {PIL.Image, np.ndarray, or list} -- A PIL image, np.ndarray, or list.

        Returns:
            PIL.Image -- If detected, cropped image of a face
                with dimensions 3 x image_size x image_size. If self.keep_all is True, n detected
                faces are returned in an n x 3 x image_size x image_size tensor. If `img` is a list
                of images, the item(s) returned have an extra dimension (batch) as the first
                dimension.

        Example:
        >>> from facenet_pytorch import MTCNN
        >>> altered_mtcnn = AlteredMTCNN()
        >>> face = altered_mtcnn(img)
        """

        # Detect faces
        with torch.no_grad():
            batch_boxes, _ = self.detect(img)

        # Determine if a batch or single image was passed
        batch_mode = True
        if not isinstance(img, (list, tuple)):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_mode = False

        # Process all bounding boxes
        faces = []
        for im, box_im in zip(img, batch_boxes):
            if box_im is None:
                faces.append(None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face = extract_face_return_results(im, box, self.image_size, self.margin)
                if self.post_process:
                    face = fixed_image_standardization(face)
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]

            faces.append(faces_im)

        if not batch_mode:
            faces = faces[0]

        return faces


def extract_face_return_results(img, box, image_size=160, margin=0):
    """Extract face + margin from PIL Image given bounding box.

        This method is nearly identical to the extract_face method provided by the MTCNN class, but
        returning the detected faces unprocessed rather than (possibly) saving them.

    Arguments:
        img {PIL.Image} -- A PIL Image.
        box {numpy.ndarray} -- Four-element bounding box.
        image_size {int} -- Output image size in pixels. The image will be square.
        margin {int} -- Margin to add to bounding box, in terms of pixels in the final image.
            Note that the application of the margin differs slightly from the davidsandberg/facenet
            repo, which applies the margin to the original image before resizing, making the margin
            dependent on the original image size.

    Returns:
        torch.tensor -- tensor representing the extracted face.
    """
    margin = [
        margin * (box[2] - box[0]) / (image_size - margin),
        margin * (box[3] - box[1]) / (image_size - margin),
    ]
    raw_image_size = get_size(img)
    box = [
        int(max(box[0] - margin[0] / 2, 0)),
        int(max(box[1] - margin[1] / 2, 0)),
        int(min(box[2] + margin[0] / 2, raw_image_size[0])),
        int(min(box[3] + margin[1] / 2, raw_image_size[1])),
    ]

    face = crop_resize(img, box, image_size)
    return face
