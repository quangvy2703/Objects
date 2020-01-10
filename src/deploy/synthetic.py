import os
import cv2
import numpy as np
import progressbar
from face_detection import FaceDetection
from face_align import FaceAlign
from objects_detection import ObjectsDetection
from scenes_change import ScenesChange
from _face_recognition import FaceRecognition, Net
from gea import GEA
from face_features import FaceFeature
import torch
from torch.autograd import Variable


prexfix_path = "/media/vy/DATA/projects/face/project3/Objects"
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

        self.emotion_prediction = None
        self.emotion_prediction_net = None
        self.showcam = True

        self.showcam = False
        self.emotion_labels = None

        if args.use_face_detection:
            self.face_detection = FaceDetection(args)
            self.face_detection.build_net()
            self.face_align = FaceAlign(args)

        if args.use_ga_prediction or args.use_emotion_prediction:
            self.gea = GEA(args)
            self.gea.build_ga_model()
            self.gea.build_emotion_model()
            self.emotion_labels = self.gea.emotion_labels

        if args.use_objects_detection:
            self.objects_detection = ObjectsDetection(args)
            self.objects_detection.build_net()

        if args.use_scenes_change_count:
            self.scenes_count = ScenesChange(args)

        if args.use_face_recognition:
            self.face_features = FaceFeature(args)
            self.face_features.build_net()
            self.face_recognition = FaceRecognition(args)
            self.face_recognition_net = Net(args)
            if not args.train:
                self.face_recognition_net.load_state_dict(torch.load(prexfix_path + "/checkpoints/checkpoint_290.pth"))

    def train_face_recognition(self):
        self.face_recognition.prepare_images(self.face_detection, self.face_align)
        self.face_recognition.augumenter(self.face_features)
        self.face_recognition.train(self.face_recognition_net)

    def run(self):
        scenes = None
        n_frames = 0
        if self.args.use_scenes_change_count:
            scenes = self.scenes_count.find_scenes()

        if self.showcam is False:
            cap = cv2.VideoCapture(self.args.input_video)
            ret, fr = cap.read()
            if fr is None:
                return
            frame_height, frame_width, _ = fr.shape

            fps = cap.get(cv2.CAP_PROP_FPS)
            n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter("video.mp4", fourcc, fps, (frame_width, frame_height))
            cmd = "ffmpeg -y -i {} -acodec libmp3lame audio.mp3".format(self.args.input_video)
            os.system(cmd)
        else:
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1200)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

        count = 0
        num_scene = 0
        if n_frames is not 0:
            bar = progressbar.ProgressBar(maxval=n_frames).start()
        while True:
            # print("Frame ", count)
            detections = []
            ret, fr = cap.read()
            if fr is None:
                break

            fr_copy = fr.copy()

            if self.args.use_face_detection:
                # fr = cv2.GaussianBlur(fr, (5, 5), cv2.BORDER_DEFAULT)
                detected_face_bboxes, cropped_faces, landmarks = self.face_detection.detection(fr)
                if self.args.use_ga_prediction:
                    for idx, _ in enumerate(cropped_faces):
                        face = self.face_align.align(fr, detected_face_bboxes[idx], landmarks[idx])
                        # cv2.imwrite("align.png", face)
                        if face is None:
                            continue

                        _outputs = [[0, 0]]
                        if self.args.use_face_recognition:
                            emb = self.face_features.run(face)
                            emb = emb.reshape(1, 512)
                            emb_v = Variable(torch.from_numpy(emb))
                            pred, _outputs = self.face_recognition.run(self.face_recognition_net, emb_v)

                        # The collision in data between get_ga and get emotion, emotion must be executed first

                        emotion = ["None", "None"]
                        prob_emotions = [1, 1, 1, 1]

                        if self.args.use_emotion_prediction:
                            emotion, prob_emotions = self.gea.get_emotion(face)

                        face = np.transpose(face, (2, 0, 1))
                        gender, age = self.gea.get_ga(face)

                        gender = "Male" if gender == 1 else "Female"
                        emotion = emotion[0]
                        if float(max(_outputs[0])) * 100 > 95:
                            object = {"bbox": detected_face_bboxes[idx], "name": [gender, age, emotion,
                                                                                  pred + "--" + str(round(
                                                                                      float(max(_outputs[0])) * 100,
                                                                                      2))],
                                      "prob_emotions":prob_emotions}
                        else:
                            object = {"bbox": detected_face_bboxes[idx], "name": [gender, age, emotion, "Unkown"],
                                      "prob_emotions": prob_emotions}
                        detections.append(object)
                else:
                    for idx, face in enumerate(detected_face_bboxes):
                        object = {"bbox": detected_face_bboxes[idx], "name": ["Face"]}
                        detections.append(object)

            if self.args.use_objects_detection:
                # print(os.getcwd())
                # cv2.imwrite("src/deploy/image.png", fr)
                boxes, scores, labels = self.objects_detection.run(fr_copy)
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    # scores are sorted so we can break
                    if score < 0.5:
                        break
                    object = {"bbox": box, "name": [self.objects_detection.labels_to_names[label]]}
                    detections.append(object)

            for _object in detections:
                cv2.rectangle(fr, (_object["bbox"][0], _object["bbox"][1]), (_object["bbox"][2], _object["bbox"][3]),
                              (255, 195, 0), 2)
                if len(_object["name"]) == 4:
                    cv2.putText(fr, "ID " + str(_object["name"][3]),
                                (_object["bbox"][0], _object["bbox"][1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (177, 18, 38), 2, cv2.LINE_AA)
                    cv2.putText(fr, str(_object["name"][1])+", "+str(_object["name"][0]),
                                (_object["bbox"][0], _object["bbox"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (177, 18, 38), 2, cv2.LINE_AA)

                    position = 0
                    color = 50
                    color1 = 0
                    if self.args.use_emotion_prediction:
                        for i in range(0, len(self.emotion_labels)):
                            cv2.putText(fr, self.emotion_labels[i], (_object["bbox"][2]+5, _object["bbox"][1] + position),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (color, color1, 50), 1, cv2.LINE_AA)

                            cv2.rectangle(fr, (_object["bbox"][2]+5, _object["bbox"][1]+position+5),
                                          (_object["bbox"][2]+5 + (int)(50*_object["prob_emotions"][0][i]), _object["bbox"][1]+position+10),
                                          (color, color1, 50), -1)

                            position += 30
                            color += 10
                            color1 += 50

                else:
                    cv2.putText(fr, str(_object["name"][0]), (_object["bbox"][0], _object["bbox"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (177, 18, 38), 2, cv2.LINE_AA)

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

            if n_frames is not 0:
                bar.update(count)
            # if count > 500:
            #     break

            if self.showcam:
                cv2.imshow("Frame", fr)
                if cv2.waitKey(32) & 0xFF == ord('q'):
                    break
            else:
                output.write(fr)

        cap.release()
        output.release()

        name, ext = os.path.splitext(self.args.output_video)
        new_name = name + "_audio" + ext
        print("save :", new_name)
        cmd = "ffmpeg -y -i {} -i {} -shortest {}".format("video.mp4",
                                                          "audio.mp3", self.args.output_video)
        os.system(cmd)
