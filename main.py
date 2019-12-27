import argparse
import os
import cv2
import numpy as np
import mxnet as mx
from easydict import EasyDict as edict

from sklearn import preprocessing
from dlib import get_frontal_face_detector, shape_predictor
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'face', 'deploy'))
import config
from retinaface import RetinaFace



sys.path.append(os.path.join(os.path.dirname(__file__), 'face', 'src', 'common'))
import face_preprocess

# Emotion
sys.path.append(os.path.abspath(os.path.join('emotion')))
from keras.models import load_model
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input

import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
_config = tf.compat.v1.ConfigProto()
_config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
_config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=_config)
set_session(sess)

###Object detection
from imageai.Detection import ObjectDetection

# Scenes change
from scenedetect.video_manager import VideoManager
from scenedetect.scene_manager import SceneManager
# For caching detection metrics and saving/loading to a stats file
from scenedetect.stats_manager import StatsManager

# For content-aware scene detection:
from scenedetect.detectors.content_detector import ContentDetector

# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

TEMPLATE = np.float32([
    (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
    (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
    (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
    (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
    (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
    (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
    (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
    (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
    (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
    (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
    (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
    (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
    (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
    (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
    (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
    (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
    (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
    (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
    (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
    (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
    (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
    (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
    (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
    (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
    (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
    (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
    (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
    (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
    (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
    (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
    (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
    (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
    (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
    (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

INV_TEMPLATE = np.float32([
    (-0.04099179660567834, -0.008425234314031194, 2.575498465013183),
    (0.04062510634554352, -0.009678089746831375, -1.2534351452524177),
    (0.0003666902601348179, 0.01810332406086298, -0.32206331976076663)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

parser = argparse.ArgumentParser(description="Synthetic")
# Face detection arguments
parser.add_argument('--use_face_detection', type=bool, default=False,
                    help="Using face detection?")
parser.add_argument('--flip', default=False, type=bool)
parser.add_argument('--face_detection_model', default='models/retinaface-R50/R50', type=str,
                    help="Path to face detection model, examples, /50 is prefix")
# parser.add_argument('--face_detection_model_path', default='../models/mnet.25/mnet.25', type=str)

# Age, emotion, gender
parser.add_argument('--use_ga_prediction', type=bool, default=False, help="Using age, emotion prediction?")
parser.add_argument('--ga_model', default='models/gamodel-r50/model, 0',
                    help='Path to gender age prediction model')
parser.add_argument('--use_emotion_prediction', type=bool, default=False,
                    help="Using emotion prediction?")
parser.add_argument('--emotion_model', type=str, default="models/emotion_model.hdf5",
                    help="Path to emotion prediction model")

# Face features
parser.add_argument('--use_face_recognition', type=bool, default=False, help="Using face recognition?")
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--emb_size', type=int, default=512, help="Embedding size of face features")
parser.add_argument('--face_feature_model', type=str, default='models/model-r100-ii/model, 0',
                    help="Path to face features extraction model")

# Object Detection
parser.add_argument('--use_objects_detection', type=bool, default=False, help="Using object detection?")
parser.add_argument('--object_detection_model', default='models/resnet50_coco_best_v2.0.1.h5',
                    help='Path to objects detection model')

# Scenes change
parser.add_argument('--use_scenes_change_count', type=bool, default=False, help="Using scenes change count")

# Input and output video
parser.add_argument('--input_video', type=str, default="", required=True, help="Path to the input video")
parser.add_argument('--output_video', type=str, default="", required=True, help="Output video name")

# GPU
parser.add_argument('--gpuid', type=int, default=0, help="GPU ID")


# Get faces in images in a given image folder
# Return file name and faces in images
class FaceDetection:
    def __init__(self, args):
        self.args = args
        self.scales = [128, 128]
        self.threshold = 0.9
        self.face_detection_model = None
        pass

    def build_net(self, net='net3'):
        """
        Building RetinaFace net
        @param net:
        @return: RetinaFace model
        """
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
        faces, landmarks = self.face_detection_model.detect(img, self.threshold, scales, args.flip)
        if len(faces) > 0:
            for i in range(faces.shape[0]):
                box = faces[i].astype(np.int)
                detected_faces.append(box)
                cropped_faces.append(img[box[1]:box[3], box[0]:box[2]])

        return detected_faces, cropped_faces, landmarks


# Align face
class FaceAlign:
    INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
    OUTER_EYES_AND_NOSE = [36, 45, 33]

    def __init__(self, face_predictor="models/shape_predictor_68_face_landmarks.dat"):
        pass

    def align(self, img, bbox, landmarks):
        nimg = face_preprocess.preprocess(img, bbox, landmarks, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        # nimg = np.transpose(nimg, (2, 0, 1))
        return nimg


# Get features of given faces in given images
class FaceFeature:
    def __init__(self, args, image_shape=[3, 112, 112]):
        self.args = args
        self.image_shape = image_shape
        self.face_feature_model = None

    def do_flip(self, data):
        for idx in range(data.shape[0]):
            data[idx, :, :] = np.fliplr(data[idx, :, :])

    def increase_brightness(self, img, threshold=170):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        value = 1.1 if brightness < threshold else 0.1

        count = 0
        while (True):
            count += 1
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            vValue = hsv[..., 2]

            hsv[..., 2] = np.where(vValue * value > 255, 255, vValue * value)
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            print(np.mean(gray))
            if np.mean(gray) > threshold and value == 1.1:
                break
            if np.mean(gray) < threshold and value == 0.1:
                break
            # if count == 20:
            #     break
        return img

    def get_gray_image(self, img):
        _img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        backtorgb = cv2.cvtColor(_img, cv2.COLOR_GRAY2RGB)
        return backtorgb

    def get_features(self, cropped_face):
        cropped_face_clone = np.array(cropped_face)

        input_blob = np.zeros((1, 3, self.image_shape[1], self.image_shape[2]))
        # For each image image login images

        # h, w, _ = face.shape
        face_rgp = np.copy(cropped_face_clone)[:, :, ::-1]

        attempts = [0]

        for flip_id in attempts:
            img_copy = np.copy(face_rgp)

            if flip_id == 1:
                self.do_flip(img_copy)

            # if self.args.norm:
            #     norm_img_copy = np.zeros((3, 112, 112))
            #     norm_img_copy = cv2.normalize(img_copy, norm_img_copy, 0, 255, cv2.NORM_MINMAX)
            #     img_copy = norm_img_copy

            img_copy = np.transpose(img_copy, (2, 0, 1))
            input_blob[0] = img_copy[:, :, ::-1]

        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.face_feature_model.model.forward(db, is_train=False)
        _embedding = self.face_feature_model.model.get_outputs()[0].asnumpy()


        if self.args.emb_size == 0:
            self.agrs.emb_size = _embedding.shape[0]

        # if self.args.use_flip:
        #     embedding1 = _embedding[0::2]
        #     embedding2 = _embedding[1::2]
        #     embedding = embedding1 + embedding2
        # else:
        print("feature size ", self.args.emb_size)
        embedding = _embedding
        embedding = preprocessing.normalize(embedding)
        return embedding

    def build_net(self):
        """
        Building features extraction network to push up get_features function
        :return: build network
        """
        ctx = []
        cvd = []
        # cvd = os.environ['CUDA_VISIBLE_DEVICES'].strip()

        ctx = [mx.gpu(self.args.gpuid)]
        # if len(cvd) > 0:
        #     ctx = [mx.gpu(i) for i in range(len(cvd.split(',')))]
        # if len(ctx) == 0:
        #     ctx = [mx.cpu()]
        #     print("use cpu")
        # else:
        #     print("gpu num: ", len(ctx))

        vec = self.args.face_feature_model.split(',')
        assert len(vec) > 1, "/path/to/mode, epoch"
        prefix = vec[0]
        epoch = int(vec[1])
        print("Loading ", prefix, " epoch ", epoch)
        net = edict()
        net.ctx = ctx
        net.sym, net.arg_params, net.aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = net.sym.get_internals()
        net.sym = all_layers['fc1_output']
        net.model = mx.mod.Module(symbol=net.sym, context=net.ctx, label_names=None)
        net.model.bind(data_shapes=[('data', (args.batch_size, 3, self.image_shape[1], self.image_shape[2]))])
        net.model.set_params(net.arg_params, net.aux_params)

        self.face_feature_model = net

    def run(self, cropped_face):
        """
        :param is_align:
        :param align:
        :param detected_faces:
        :param net: features extraction network
        :return: a features array
        """

        embedding = self.get_features(cropped_face)

        return embedding


# Gender emotion and age
class GEA:
    def __init__(self, args):
        # _config = tf.compat.v1.ConfigProto()
        # _config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        # _config.log_device_placement = True  # to log device placement (on which device the operation ran)
        # sess = tf.compat.v1.Session(config=_config)
        # set_session(sess)

        self.args = args
        self.ctx = mx.gpu(args.gpuid)

        # Age Gender
        self.image_size = (112, 112)
        self.layer = 'fc1'
        self.ga_model = None

        # Emotion
        self.emotion_model = None
        self.emotion_labels = None
        emotion_offsets = (20, 40)
        self.emotion_target_size = None
        pass

    def build_ga_model(self):
        _vec = self.args.ga_model.split(',')
        assert len(_vec) == 2
        prefix = _vec[0]
        epoch = int(_vec[1])
        print('loading', prefix, epoch)
        sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
        all_layers = sym.get_internals()
        sym = all_layers[self.layer + '_output']
        model = mx.mod.Module(symbol=sym, context=self.ctx, label_names=None)
        model.bind(data_shapes=[('data', (1, 3, self.image_size[0], self.image_size[1]))])
        model.set_params(arg_params, aux_params)

        self.ga_model = model

    def build_emotion_model(self):
        self.emotion_model = load_model(self.args.emotion_model)
        self.emotion_target_size = self.emotion_model.input_shape[1:3]
        self.emotion_labels = get_labels('fer2013')

    def get_ga(self, img):
        input_blob = np.expand_dims(img, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.ga_model.forward(db, is_train=False)
        ret = self.ga_model.get_outputs()[0].asnumpy()
        g = ret[:, 0:2].flatten()
        gender = np.argmax(g)
        a = ret[:, 2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))

        return gender, age

    # {0:'angry',1:'disgust',2:'fear',3:'happy',4:'sad',5:'surprise',6:'neutral'}
    def get_emotion(self, img):
        emotion_text = []
        emotion_probability = []
        gray_face = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.resize(gray_face, self.emotion_target_size)
        gray_face = preprocess_input(gray_face, True)
        gray_face = np.expand_dims(gray_face, 0)
        gray_face = np.expand_dims(gray_face, -1)
        emotion_prediction = self.emotion_model.predict(gray_face)
        emotion_probability.append(np.max(emotion_prediction))
        emotion_label_arg = np.argmax(emotion_prediction)
        emotion_text.append(self.emotion_labels[emotion_label_arg])

        return emotion_text, emotion_probability


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
        # (name, ext) = self.args.output_video.rsplit('/')[-1].rsplit('.', 2)
        # cap = cv2.VideoCapture(self.args.output_video)
        # ret, fr = cap.read()
        # if fr is None:
        #     return
        #
        # frame_height, frame_width, _ = fr.shape
        #
        # fps = cap.get(cv2.CAP_PROP_FPS)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter(self.args.output_video.rsplit('.', 2)[0] + "_output." + ext,
        #                       fourcc, fps, (frame_width, frame_height))
        #
        # while True:
        #     ret, fr = camera.read()
        #     if fr is None:
        #         break
        #     fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        #     out.write(fr)


class FaceRecognition:
    def __init__(self, args):
        self.args = args
        self.align = FaceAlign()
        self.features = FaceFeature(args)
        self.features_net = FaceFeature(args).build_net()
        self.face_detection = FaceDetection(args)
        self.face_detection_net = self.face_detection.build_net()

    def get_face_features(self, faces):
        return self.features.run(self.features_net, faces, self.align)

    def get_faces(self, img):
        face_boxes, cropped_faces, landmarks = self.face_detection.detection(img, self.face_detection_net)
        return face_boxes, cropped_faces, landmarks

    def face_distance(self, face_encodings, face_to_compare):
        """
        Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
        for each comparison face. The distance tells you how similar the faces are.
        :param faces: List of face encodings to compare
        :param face_to_compare: A face encoding to compare against
        :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
        """
        if len(face_encodings) == 0:
            return np.empty((0))

        return np.linalg.norm(face_encodings - face_to_compare, axis=1)

    def compare_faces(self, known_face_encodings, face_encoding_to_check, tolerance=0.1):
        """
        Compare a list of face encodings against a candidate encoding to see if they match.
        :param known_face_encodings: A list of known face encodings
        :param face_encoding_to_check: A single face encoding to compare against the list
        :param tolerance: How much distance between faces to consider it a match. Lower is more strict. 0.6 is typical best performance.
        :return: A list of True/False values indicating which known_face_encodings match the face encoding to check
        """
        distance = self.face_distance(known_face_encodings, face_encoding_to_check)
        return distance


class ScenesChange:
    def __init__(self, args):
        self.args = args

        pass

    def find_scenes(self):
        # type: (str) -> List[Tuple[FrameTimecode, FrameTimecode]]
        video_manager = VideoManager([self.args.input_video])
        stats_manager = StatsManager()
        # Construct our SceneManager and pass it our StatsManager.
        scene_manager = SceneManager(stats_manager)

        # Add ContentDetector algorithm (each detector's constructor
        # takes detector options, e.g. threshold).
        scene_manager.add_detector(ContentDetector())
        base_timecode = video_manager.get_base_timecode()

        scene_list = []
        results = []

        try:

            # Set downscale factor to improve processing speed.
            video_manager.set_downscale_factor()

            # Start video_manager.
            video_manager.start()

            # Perform scene detection on video_manager.
            scene_manager.detect_scenes(frame_source=video_manager)

            # Obtain list of detected scenes.
            scene_list = scene_manager.get_scene_list(base_timecode)
            # Each scene is a tuple of (start, end) FrameTimecodes.

            print('List of scenes obtained:')
            for i, scene in enumerate(scene_list):
                print(
                    'Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
                        i + 1,
                        scene[0].get_timecode(), scene[0].get_frames(),
                        scene[1].get_timecode(), scene[1].get_frames(),))

            print('List of scenes obtained:')
            for i, scene in enumerate(scene_list):
                if (scene[1].get_frames() - scene[0].get_frames()) > 10:
                    results.append([scene[0].get_frames(), scene[1].get_frames()])

        finally:
            video_manager.release()

        return results



class Synthetic:
    def __init__(self, args):
        self.args = args
        self.face_detection = None
        self.face_align = None
        self.gea = None
        self.objects_detection = None
        self.scenes_count = None
        self.face_features = None
        # self.showcam = True
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


    def run(self):
        scenes = None
        if self.args.use_scenes_change_count:
            scenes = self.scenes_count.find_scenes()

        if self.args.use_objects_detection or self.args.use_scenes_change_count:
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
                print("Frame is None")
                break

            if self.args.use_face_detection:
                print("Face Detection ", count)
                detected_face_bboxes, cropped_faces, landmarks = self.face_detection.detection(fr)
                if self.args.use_ga_prediction:
                    for idx, _ in enumerate(cropped_faces):
                        face = self.face_align.align(fr, detected_face_bboxes[idx], landmarks[idx])
                        # cv2.imwrite("align.png", face)
                        if face is None:
                            continue
                        if self.args.use_face_recognition:
                            face_features = self.face_features.run(face)
                        # The colision in data between get_ga and get emotion, emotion must be executed first
                        emotion = self.gea.get_emotion(face)
                        face = np.transpose(face, (2, 0, 1))
                        gender, age = self.gea.get_ga(face)

                        gender = "Male" if gender == 1 else "Female"
                        emotion = emotion[0]
                        object = {"bbox": detected_face_bboxes[idx], "name": [gender, age, emotion]}
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
                if len(_object["name"]) == 3:
                    cv2.putText(fr, "Emotion " + str(_object["name"][2][0]), (_object["bbox"][0], _object["bbox"][1] - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                    cv2.putText(fr, "Gender " + str(_object["name"][0]), (_object["bbox"][0], _object["bbox"][1] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                    cv2.putText(fr, "Age " + str(_object["name"][1]), (_object["bbox"][0], _object["bbox"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(fr, str(_object["name"][0]), (_object["bbox"][0], _object["bbox"][1]),
                                cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 195, 0), 2, cv2.LINE_AA)

            if (self.args.use_objects_detection is False and self.args.use_scenes_change_count is False) or self.showcam:
                cv2.imshow("Frame", fr)
                if cv2.waitKey(32) & 0xFF == ord('q'):
                    break
            if args.use_scenes_change_count:
                if (count < scenes[num_scene][1]):
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
            count += 1
            if self.args.use_objects_detection or self.args.use_scenes_change_count:
                output.write(fr)


def main(args):
    syn = Synthetic(args)
    syn.run()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)
