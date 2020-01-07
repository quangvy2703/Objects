import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'deploy'))
import config

from synthetic import Synthetic
import argparse

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
# parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--emb_size', type=int, default=512, help="Embedding size of face features")
parser.add_argument('--face_feature_model', type=str, default='models/model-r100-ii/model, 0',
                    help="Path to face features extraction model")

# Object Detection
parser.add_argument('--use_objects_detection', type=bool, default=False, help="Using object detection?")
parser.add_argument('--object_detection_model', default='models/resnet50_coco_best_v2.0.1.h5',
                    help='Path to objects detection model')

# Face recognition

parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--epochs', type=int, default=300, help="Epochs")
# parser.add_argument('--emb_size', type=int, default=512, help="Embedding size")
parser.add_argument('--train_dir', type=str, default="datasets/train/features")
parser.add_argument('--test_dir', type=str, default="datasets/val/features")
parser.add_argument('--n_classes', type=int, default=3, help="Number of peoples")
parser.add_argument('--train', action='store_true')

# Scenes change
parser.add_argument('--use_scenes_change_count', type=bool, default=False, help="Using scenes change count")

# Input and output video
parser.add_argument('--input_video', type=str, default="", required=True, help="Path to the input video")
parser.add_argument('--output_video', type=str, default="", required=True, help="Output video name")

# GPU
parser.add_argument('--gpuid', type=int, default=0, help="GPU ID")


def main(args):
    syn = Synthetic(args)
    if args.train:
        # syn.train_face_recognition()
        syn.train_emotion_prediction()
    syn.run()


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    main(args)


