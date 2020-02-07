import cv2
import mxnet as mx
import numpy as np
from keras.models import load_model
from utils.datasets import get_labels
from utils.preprocessor import preprocess_input
import tensorflow as tf
import tensorflow.python.keras.backend as K
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # use id from $ nvidia-smi

class GEA:
    def __init__(self, args):
        _config = tf.compat.v1.ConfigProto()
        _config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        _config.log_device_placement = True  # to log device placement (on which device the operation ran)
        sess = tf.compat.v1.Session(config=_config)
        K.set_session(sess)

#        num_cores = 8
#        GPU = False
#        CPU = True
#        if GPU:
#            num_GPU = 1
#            num_CPU = 1
#        if CPU:
#            num_CPU = 1
#            num_GPU = 0

#        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
#                                inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
#                                device_count={'CPU': num_CPU, 'GPU': num_GPU})
#        session = tf.Session(config=config)
#        K.set_session(session)
        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
        # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        # K.set_session(sess)

        self.args = args
        self.ctx = mx.gpu(args.gpuid)
#        self.ctx = mx.cpu()

        # Age Gender
        self.image_size = (112, 112)
        self.layer = 'fc1'
        self.ga_model = None

        # Emotion
        self.emotion_model = None
        self.emotion_labels = None
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
        with tf.device('/cpu:0'):
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
        with tf.device('/cpu:0'):
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

            return emotion_text, emotion_prediction
