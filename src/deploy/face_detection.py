# Get faces in images in a given image folder
# Return file name and faces in images

import numpy as np
from retinaface import RetinaFace

class FaceDetection:
    def __init__(self, args):
        self.args = args
        self.scales = [1024, 1980]
        self.threshold = 0.9
        self.face_detection_model = None
        pass

    def build_net(self, net='net3'):
        """
        Building RetinaFace net
        @param net:
        @return: RetinaFace model
        """

        # self.face_detection_model = RetinaFace(self.args.face_detection_model, 0, self.args.gpuid, net)
        self.face_detection_model = RetinaFace(self.args.face_detection_model, 0, self.args.gpuid, net)

    def detection(self, img):
        """
        Detect faces in given image
        @param img: input image
        @param face_detection_net:
        @return:
            detected_faces: face boxes in the image
            crooped_faces: cropped image faces from image
            landmarks: landmark points
        """
        detected_faces = []
        cropped_faces = []

        img_shape = img.shape
        target_size = self.scales[0]
        max_size = self.scales[1]
        img_size_min = np.min(img_shape[:2])
        img_size_max = np.max(img_shape[:2])
        img_scale = 1.0 * target_size / img_size_min
        if np.round(img_scale * img_size_max) > max_size:
            img_scale = 1.0 * max_size / img_size_max

        scales = [img_scale]
        faces, landmarks = self.face_detection_model.detect(img, self.threshold, scales, self.args.flip)
        if len(faces) > 0:
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                detected_faces.append(box)
                cropped_faces.append(img[box[1]:box[3], box[0]:box[2]])

        return detected_faces, cropped_faces, landmarks
