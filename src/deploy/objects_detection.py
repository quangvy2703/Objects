from imageai.Detection import ObjectDetection

class ObjectsDetection:
    def __init__(self, args):
        self.args = args
        self.detector = ObjectDetection()
        self.detector.setModelTypeAsRetinaNet()
        self.detector.setModelPath(args.object_detection_model)
        self.detector.loadModel()

    def run(self, img):
        returned_image, detections = self.detector.detectObjectsFromImage(input_image=img, input_type="array",
                                                                          output_type="array",
                                                                          minimum_percentage_probability=40)

        return returned_image, detections
