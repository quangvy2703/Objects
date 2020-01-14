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
COLOR = (37, 117, 17)

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

        if args.use_gender_prediction or args.use_emotion_prediction or args.use_age_prediction:
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

        count = 1
        num_scene = 0
        if n_frames is not 0:
            bar = progressbar.ProgressBar(maxval=n_frames).start()
        while True:
            # print("Frame ", count)
            detections = []
            ret, fr = cap.read()
            if fr is None:
                break

            fr_draw = fr.copy()

            if self.args.use_face_detection:
                # fr = cv2.GaussianBlur(fr, (5, 5), cv2.BORDER_DEFAULT)
                detected_face_bboxes, cropped_faces, landmarks = self.face_detection.detection(fr)

                for idx, _ in enumerate(cropped_faces):
                    box = detected_face_bboxes[idx]
                    y_pos = 0
                    face = self.face_align.align(fr, box, landmarks[idx])
                    face_clone = face.copy()
                    # cv2.imwrite("align.png", face)
                    if face is None:
                        continue

                    # Draw face box
                    cv2.rectangle(fr_draw, (box[0], box[1]),(box[2], box[3]), (255, 195, 0), 2)

                    if self.args.use_emotion_prediction:
                        emotion, prob_emotions = self.gea.get_emotion(face)
                        emotion = emotion[0]

                        position = 10
                        color = 50
                        color1 = 0
                        space = box[3] - box[1]
                        font = space / (15 * 30)
                        # print(font)
                        text_size = cv2.getTextSize("text", cv2.FONT_HERSHEY_SIMPLEX, font, 1)[0][1]

                        for i in range(0, len(self.emotion_labels)):
                            cv2.putText(fr_draw, self.emotion_labels[i],
                                        (box[2] + 5, box[1] + position),
                                        cv2.FONT_HERSHEY_SIMPLEX, font, (color, color1, 50), 1, cv2.LINE_AA)
                            cv2.rectangle(fr_draw, (box[2] + 5, box[1] + position + text_size),
                                          (box[2] + 5 + (int)(50 * prob_emotions[0][i]),
                                           box[1] + position + text_size + 5),
                                          (color, color1, 50), -1)

                            position = position + text_size + int(font * 40)
                            color += 10
                            color1 += 50


                        # color = 50
                        # color1 = 0
                        # h_box = box[3] - box[1]
                        # cell = round(h_box / 7)
                        # cell = int(cell)
                        # position = 0
                        # for i in range(0, len(self.emotion_labels)):
                        #     cv2.rectangle(fr_draw, (box[2] + 5, box[1] + position + 5),
                        #                   (box[2] + 5 + (int)(50 * prob_emotions[0][i]), box[1] + position),
                        #                   (90, 207, 189), -1)
                        #     cv2.putText(fr_draw, self.emotion_labels[i], (box[2] + 5, box[1] + position),
                        #                 cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                        #
                        #
                        #     position += cell
                        #     color += 10
                        #     color1 += 50

                    if self.args.use_gender_prediction or self.args.use_age_prediction:
                        face = np.transpose(face, (2, 0, 1))
                        gender, age = self.gea.get_ga(face)

                        gender = "Male" if gender == 1 else "Female"


                        # Draw gender
                        if self.args.use_gender_prediction:
                            cv2.putText(fr_draw, gender, (box[0], box[1] - y_pos * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, COLOR, 1, cv2.LINE_AA)
                            y_pos += 1

                        # Draw age
                        if self.args.use_age_prediction:
                            cv2.putText(fr_draw, "Age " + str(age), (box[0], box[1] - y_pos * 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, COLOR, 1, cv2.LINE_AA)
                            y_pos += 1

                    if self.args.use_face_recognition:
                        emb = self.face_features.run(face_clone)
                        emb = emb.reshape(1, 512)
                        emb_v = Variable(torch.from_numpy(emb))
                        pred, _outputs = self.face_recognition.run(self.face_recognition_net, emb_v)

                        # Draw ID
                        if float(max(_outputs[0])) * 100 > 90:
                            probability = str(round(float(max(_outputs[0])) * 100, 2))
                            cv2.putText(fr_draw, "ID -- " + pred + ' -- ' + str(probability),
                                        (box[0], box[1] - y_pos*20),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, COLOR, 1, cv2.LINE_AA)
                        else:
                            cv2.putText(fr_draw, "ID -- Unknown", (box[0], box[1] - y_pos*20),
                                        cv2.FONT_HERSHEY_SIMPLEX, .5, COLOR, 1, cv2.LINE_AA)
                            y_pos += 1

                        # The collision in data between get_ga and get emotion, emotion must be executed first

            if self.args.use_objects_detection:
                y_pos = 0
                boxes, scores, labels = self.objects_detection.run(fr)
                for box, score, label in zip(boxes[0], scores[0], labels[0]):
                    box = box.astype(int)

                    # scores are sorted so we can break
                    if score < 0.5:
                        break
                    # print(box)
                    # print((box[0], box[1]), (box[2], box[3]))
                    cv2.rectangle(fr_draw, (box[0], box[1]), (box[2], box[3]), (255, 195, 0), 2)
                    name = self.objects_detection.labels_to_names[label]
                    cv2.putText(fr_draw, name, (box[0], box[1] - y_pos * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (177, 18, 38), 2, cv2.LINE_AA)

            if self.args.use_scenes_change_count:
                if count < scenes[num_scene][1]:
                    C = num_scene
                else:
                    if num_scene < len(scenes) - 1:
                        num_scene += 1
                cv2.putText(
                    img=fr_draw,
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
                cv2.imshow("Frame", fr_draw)
                if cv2.waitKey(32) & 0xFF == ord('q'):
                    break
            else:
                output.write(fr_draw)

        cap.release()
        output.release()

        name, ext = os.path.splitext(self.args.output_video)
        new_name = name + "_audio" + ext
        print("save :", new_name)
        cmd = "ffmpeg -y -i {} -i {} -shortest {}".format("video.mp4",
                                                          "audio.mp3", self.args.output_video)
        os.system(cmd)
