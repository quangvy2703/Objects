import numpy as np
import mxnet as mx
from sklearn import preprocessing
from easydict import EasyDict as edict
import os
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use id from $ nvidia-smi

# Get features of given faces in given images
class FaceFeature:
    def __init__(self, args, image_shape=[3, 112, 112]):
        self.args = args
        self.image_shape = image_shape
        self.face_feature_model = None

    def do_flip(self, data):
        for idx in range(data.shape[0]):
            data[idx, :, :] = np.fliplr(data[idx, :, :])

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

            img_copy = np.transpose(img_copy, (2, 0, 1))
            input_blob[0] = img_copy[:, :, ::-1]

        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,))
        self.face_feature_model.model.forward(db, is_train=False)
        _embedding = self.face_feature_model.model.get_outputs()[0].asnumpy()

        if self.args.emb_size == 0:
            self.agrs.emb_size = _embedding.shape[0]

        # print("feature size ", self.args.emb_size)
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
        net.model.bind(data_shapes=[('data', (self.args.batch_size, 3, self.image_shape[1], self.image_shape[2]))])
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


