import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim.lr_scheduler import CyclicLR
import imgaug as ia
from imgaug import augmenters as iaa
import cv2
import torch
from torch.autograd import Variable
import tqdm

categories = {'Ha-Lan': 0, 'Ngan': 1, 'Dung': 2, 'Tra-Long': 3}
hidden_dim = 2048
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FacialDataLoader(DataLoader):
    def __init__(self, args):
        self.args = args
        self.data_x = np.array([], dtype=np.float32).reshape(0, args.emb_size)
        self.data_y = None

        self.transform = transform
        pass

    def __len__(self):
        return len(self.data_x)

    def load_data(self):
        characters = os.listdir(self.args.train_dir + "/../features")
        for character in characters:
            print("Loading data ", character.rsplit('.', 2), categories[character.rsplit('.', 2)[0]])
            features = np.load(os.path.join(self.args.train_dir, "..", "features", character))
            self.data_x = np.concatenate((self.data_x, features))
            if self.data_y is None:
                self.data_y = np.array([categories[character.rsplit('.', 2)[0]]] * len(features))
            else:
                self.data_y = np.concatenate((self.data_y,
                                            np.array([categories[character.rsplit('.', 2)[0]]] * len(features))))


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {"features": self.data_x[idx, :], "character": self.data_y[idx]}
        #
        # if self.transform:
        #     sample = self.transform(sample)

        return sample

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.emb_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, args.n_classes)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = F.softmax(out, dim=1)
        return out


class FaceRecognition:
    def __init__(self, args):
        self.args = args
        pass

    def prepare_images(self, detection, align):
        train_images = os.listdir(self.args.train_dir)
        # test_images = os.listdir(args.test_dir)

        if not os.path.exists(os.path.join(self.args.train_dir, "..", "aligned")):
            os.makedirs(os.path.join(self.args.train_dir, "..", "aligned"))

        for person in train_images:
            person_imgs = os.listdir(os.path.join(self.args.train_dir, person))

            if not os.path.exists(os.path.join(self.args.train_dir, "..", "aligned", person)):
                os.makedirs(os.path.join(self.args.train_dir, "..", "aligned", person))

            for train_image in person_imgs:
                img_path = os.path.join(self.args.train_dir, person, train_image)
                print(img_path)
                img = cv2.imread(img_path)
                detected_face_bboxes, cropped_faces, landmarks = detection.detection(img)

                for idx, cropped_face in enumerate(cropped_faces):
                    aligned_path = os.path.join(self.args.train_dir, "..", "aligned", person, str(idx) + "_" + train_image)
                    print(aligned_path)
                    face = align.align(img, detected_face_bboxes[idx], landmarks[idx])

                    cv2.imwrite(aligned_path, face)

    def prewhiten(self, x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1 / std_adj)
        return y

    def augumenter(self, features_extraction):
        sometimes = lambda aug: iaa.Sometimes(0.8, aug)
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            sometimes(
                iaa.OneOf([
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    iaa.AddToHueAndSaturation((-20, 20)),
                    iaa.Add((-20, 20), per_channel=0.5),
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.GaussianBlur((0, 2.0)),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Sharpen(alpha=(0, 0.5), lightness=(0.7, 1.3)),
                    iaa.Emboss(alpha=(0, 0.5), strength=(0, 1.5))
                ])
            )
        ])

        images_dir = os.path.join(self.args.train_dir, "..", "aligned")
        characeters = os.listdir(images_dir)
        print("Augumenting data...")
        for character in characeters:
            features = []
            images = os.listdir(os.path.join(images_dir, character))
            for image in images:
                image_path = os.path.join(images_dir, character, image)
                img = cv2.imread(image_path)
                for i in range(100):
                    img_aug = seq.augment_image(img)
                    # img_aug = self.prewhiten(img_aug)

                    embed = features_extraction.run(img_aug)
                    features.append(embed)

            save_path = os.path.join(self.args.train_dir, "..", "features", character + ".npy")
            if not os.path.exists(os.path.join(self.args.train_dir, "..", "features")):
                os.mkdir(os.path.join(self.args.train_dir, "..", "features"))
            features = np.asarray(features).reshape((len(features), self.args.emb_size))
            print(character, features.shape)
            np.save(save_path, features)


    def train(self, net):
        net.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(net.parameters(), lr=1e-3)
        scheduler = CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-3,
                                                mode='exp_range', cycle_momentum=True)
        train_data = FacialDataLoader(self.args)
        train_data.load_data()

        trainloader = torch.utils.data.DataLoader(train_data, batch_size=self.args.batch_size,
                                                  shuffle=True, num_workers=2)

        for epoch in range(self.args.epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data["features"], data["character"]
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                output = net(inputs)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                scheduler.step()

            print('[%d] loss: %.6f' %
                  (epoch, running_loss))

            if epoch % 10 == 0 or epoch == self.args.epochs - 1:
                torch.save(net.state_dict(), 'checkpoints/checkpoint_' + str(epoch) + '.pth')
        print('Finished Training')

    def run(self, net, img):
        correct = 0
        total = 0
        with torch.no_grad():
            outputs = net(img)
            _, predicted = torch.max(outputs.data, 1)

        return list(categories.keys())[predicted], outputs.detach()

    def test_on_video(self, net, detection, align, features, checkpoint_path):
        net.load_state_dict(torch.load(checkpoint_path))
        video = self.args.video_test
        cap = cv2.VideoCapture(video)
        while True:
            ret, fr = cap.read()
            faces, cropped_faces, landmarks = detection.detection(fr)
            for idx, cropped_face in enumerate(cropped_faces):
                face = align.align(fr, faces[idx], landmarks[idx])
                emb = features.run(face)
                emb = emb.reshape(1, 512)
                emb_v = Variable(torch.from_numpy(emb))
                pred, outputs = self.run(net, emb_v)

                cv2.rectangle(fr, (faces[idx][0], faces[idx][1]), (faces[idx][2], faces[idx][3]), (255, 195, 0), 2)
                cv2.putText(fr, pred + ' -- ' + str(max(outputs) * 100), (faces[idx][0], faces[idx][1]),
                            cv2.FONT_HERSHEY_COMPLEX, .5,
                            (255, 195, 0), 2, cv2.LINE_AA)

                cv2.imshow("Testing", fr)
