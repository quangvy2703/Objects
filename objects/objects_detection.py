import argparse
import os
import cv2
from imageai.Detection import ObjectDetection

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"


parser = argparse.ArgumentParser(description="Login model")
parser.add_argument('--object_detection_model', default='../models/resnet50_coco_best_v2.0.1.h5', help='path to load model.')

gpuid = 0
thresh = 0.9
box_shape = (3, 112, 122)


if __name__ == '__main__':
    args = parser.parse_args()
    execution_path = os.getcwd()

    camera = cv2.VideoCapture(0)
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath(args.object_detection_model)
    detector.loadModel()

    # custom = detector.CustomObjects(person=True, dog=True, bench=True)

    img = cv2.imread("image2.jpg")

    returned_image, detections = detector.detectObjectsFromImage(input_image=img, input_type="array",
                                                                 output_type="array",
                                                                 minimum_percentage_probability=30)

    print(detections)
    print(type(returned_image))
    cv2.imwrite("detected.png", returned_image)