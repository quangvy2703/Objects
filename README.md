


### Objects detection + face detection + face recognition + gender prediction + age prediction + emotion prediction + scenes change.

By Vy Pham & Liem Truong (KnG)

![alt text](https://www.upsieutoc.com/images/2020/01/02/Screenshot-from-2020-01-02-16-05-36.png)

## Requirements
```Shell
  Pytorch 1.1.0
  Tensorflow 1.15
  Keras 2.3.1
  imageai
  sklearn 0.22
  mxnet 1.5.1
  skimage 0.16.2
  imgaug 0.3.0
  scenedetect 0.5.1
```

### Train face recognition

1. Push your dataset as
```Shell
    datasets/
       train/
          image-folder-p1/
          image-folder-p2/
            ...
          image-folder-pn/
```

2. You can refer to the example code in ``src\deploy\synthetic.py``.
```Shell
  Line 47.  self.face_recognition_net.load_state_dict(torch.load("checkpoints/checkpoint_290.pth")) # Load face recognition model
  Line 50. self.face_recognition.prepare_images(self.face_detection, self.face_align) # Cropped faces from images
  # After that ./datasets/train/aligned/ contain faces cropped images from training datasets. Check and remove wrong faces.
  Line 51. self.face_recognition.augumenter(self.face_features) # Extracting features and saving into datasets/train/features/
  Line 52. self.face_recognition.train(self.face_recognition_net) # Starting training.
```

3. Models will be saved at ./checkpoints/

## Optionals
Change your option in ```main.py```
```
[Vy Pham](quangvy2703@gmail.com)
[Liem Truong](truongthithuyliem@gmail.com)
```
