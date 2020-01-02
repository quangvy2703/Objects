import cv2
import numpy as np
from face_detection import FaceDetection
from face_align import FaceAlign
from objects_detection import ObjectsDetection
from scenes_change import ScenesChange
from face_recognition import FaceRecognition, Net
from gea import GEA
from face_features import FaceFeature
import torch
from torch.autograd import Variable


class Synthetic:
    def __init__(self, args):
        self.args = args
        self.face_detection = None
        self.face_align = None
        self.gea = None
        self.objects_detection = None
        self.scenes_count = None
        self.face_features = None
        self.face_recognition_net = None
        self.showcam = False
        if args.use_face_detection:
            self.face_detection = FaceDetection(args)
            self.face_detection.build_net()
            self.face_align = FaceAlign(args)

        if args.use_ga_prediction or args.use_emotion_prediction:
            self.gea = GEA(args)
            self.gea.build_ga_model()
            self.gea.build_emotion_model()

        if args.use_objects_detection:
            self.objects_detection = ObjectsDetection(args)

        if args.use_scenes_change_count:
            self.scenes_count = ScenesChange(args)

        if args.use_face_recognition:
            self.face_features = FaceFeature(args)
            self.face_features.build_net()
            self.face_recognition = FaceRecognition(args)
            self.face_recognition_net = Net(args)
            if not args.train:
                self.face_recognition_net.load_state_dict(torch.load("checkpoints/checkpoint_290.pth"))

    def train_face_recognition(self):
        self.face_recognition.prepare_images(self.face_detection, self.face_align)
        self.face_recognition.augumenter(self.face_features)
        self.face_recognition.train(self.face_recognition_net)

    def run(self):
        scenes = None
        if self.args.use_scenes_change_count:
            scenes = self.scenes_count.find_scenes()

        if (self.args.use_objects_detection or self.args.use_scenes_change_count) or self.showcam is False:
            cap = cv2.VideoCapture(self.args.input_video)
            ret, fr = cap.read()
            if fr is None:
                return
            frame_height, frame_width, _ = fr.shape

            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter(self.args.output_video, fourcc, fps, (frame_width, frame_height))
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        count = 0
        num_scene = 0
        while True:
            # print("Frame ", count)
            detections = []
            ret, fr = cap.read()
            if fr is None:
                break

            if self.args.use_face_detection:
                # fr = cv2.GaussianBlur(fr, (5, 5), cv2.BORDER_DEFAULT)
                detected_face_bboxes, cropped_faces, landmarks = self.face_detection.detection(fr)
                if self.args.use_ga_prediction:
                    for idx, _ in enumerate(cropped_faces):
                        face = self.face_align.align(fr, detected_face_bboxes[idx], landmarks[idx])
                        cv2.imwrite("align.png", face)
                        if face is None:
                            continue
                        if self.args.use_face_recognition:
                            emb = self.face_features.run(face)
                            emb = emb.reshape(1, 512)
                            emb_v = Variable(torch.from_numpy(emb))
                            pred, _outputs = self.face_recognition.run(self.face_recognition_net, emb_v)

                        # The collision in data between get_ga and get emotion, emotion must be executed first
                        emotion = self.gea.get_emotion(face)
                        face = np.transpose(face, (2, 0, 1))
                        gender, age = self.gea.get_ga(face)

                        gender = "Male" if gender == 1 else "Female"
                        emotion = emotion[0]
                        if float(max(_outputs[0]))*100 > 80:
                            object = {"bbox": detected_face_bboxes[idx], "name": [gender, age, emotion,
                                                                                  pred + "--" + str(round(float(max(_outputs[0]))*100, 2))]}
                        else:
                            object = {"bbox": detected_face_bboxes[idx], "name": [gender, age, emotion, "Unkown"]}
                        detections.append(object)
                else:
                    for idx, face in enumerate(detected_face_bboxes):
                        object = {"bbox": detected_face_bboxes[idx], "name": ["Face"]}
                        detections.append(object)

            if self.args.use_objects_detection:
                returned_image, _detections = self.objects_detection.run(fr)
                for _object in _detections:
                    object = {"bbox": _object["box_points"], "name": [_object["name"]]}
                    detections.append(object)

            for _object in detections:
                print(_object)
                cv2.rectangle(fr, (_object["bbox"][0], _object["bbox"][1]), (_object["bbox"][2], _object["bbox"][3]),
                              (255, 195, 0), 2)
                if len(_object["name"]) == 4:
                    cv2.putText(fr, "ID " + str(_object["name"][3]),
                                (_object["bbox"][0], _object["bbox"][1] - 60),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                    cv2.putText(fr, "Emotion " + str(_object["name"][2][0]),
                                (_object["bbox"][0], _object["bbox"][1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                    cv2.putText(fr, "Gender " + str(_object["name"][0]), (_object["bbox"][0], _object["bbox"][1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                    cv2.putText(fr, "Age " + str(_object["name"][1]), (_object["bbox"][0], _object["bbox"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(fr, str(_object["name"][0]), (_object["bbox"][0], _object["bbox"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)

            if self.args.use_scenes_change_count:
                if count < scenes[num_scene][1]:
                    C = num_scene
                else:
                    num_scene += 1
                cv2.putText(
                    img=fr,
                    text="Scene : " + str(C),
                    org=(30, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1.0,
                    color=(255, 0, 255))
                # output.write(fr)
                count += 1

            if self.showcam:
                cv2.imshow("Frame", fr)
                if cv2.waitKey(32) & 0xFF == ord('q'):
                    break
            else:
                output.write(fr)

