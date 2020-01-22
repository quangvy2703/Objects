.
├── checkpoints			| Trained face recognition models
├── datasets
│   ├── aligned
│   │   ├── Dung		|  Cropped face images of each person
│   │   ├── Ha-Lan		|  You must delete unclear images, and 
│   │   ├── Ngan		|  wrong cropped face images
│   │   └── Tra-Long		|
│   ├── features
│   │   ├── Dung.npy
│   │   ├── Ha-Lan.npy
│   │   ├── Ngan.npy
│   │   └── Tra-Long.npy
│   └── train
│       ├── Dung		|
│       │   ├── 1.jpg		|	
│       │   ├── ....		|
│       │   └── 30.png		|
│       ├── Ha-Lan		|  Put training data here
│       │   ├── 1.jpg		|  Each folder contain images of each person
│       │   ├── ....		|  In this examples, we have 4 persons
│       │   └── 30.png		|
│       ├── Ngan		|
│       │   ├── 1.jpg		|
│       │   ├── ....		|
│       │   └── 30.png		|
│       └── Tra-Long		|
│           ├── 1.jpg		|
│           ├── ....		|
│           └── 30.png		|
├── main.py
├── models
├── README.md
├── run.sh			|  Training bash file
├── src				|  Source code in this folder
│   ├── deploy			|
│   │   ├── config.py		|
│   │   ├── config.pyc		|
│   │   ├── define.py		|
│   │   ├── face_align.py	|
│   │   ├── face_detection.py	|
│   │   ├── face_features.py	|
│   │   ├── _face_recognition.py|  Line 19 in this file, change the categories according to your dataset 
│   │   ├── gea.py		|
│   │   ├── helper.py		| 
│   │   ├── __init__.py		|
│   │   ├── lib.py		|	
│   │   ├── objects_detection.py|
│   │   ├── rcnn		|
│   │   ├── retinaface.py	|
│   │   ├── scenes_change.py	|
│   │   ├── synthetic.py	|  Line 15, change prefix_path according to your computer
│   │   ├── synthetic.pyc	|
│   │   └── utils		|
│   ├── __init__.py
│   ├── LICENSE
│   ├── README.md
│   └── src
└── videos
    └── results			| Processed videos dir


The details of run.sh file to train with your above config
	python main.py   --input_video videos/video_ob.mp4 \
		         --output_video videos/results/video_ob.mp4 \
		         --train_dir datasets/train \
		         --test_dir datasets/test  \
		         --use_face_detection True \
		         --use_face_recognition True \
		         --emb_size 512 \
		         --n_classes 4 \	# Number of peoples in your datasets
		         --train   \

Step 1. Edit the function `train_face_recognition` in `systhetic.py` file [line 63-65]
	Step 1.1. Comment line 64, 65 to crop face from training images. Run `bash run.sh`. After success, you should check the cropped faces in `datasets/aligned` to delete unclear faces, and wrong faces.
	Step 1.2. Comment line 63, uncomment line 64, 65 and re-run `bash run.sh` to start training.
Step 2. After training success, trained model located in the `checkpoints` folder. Starting server and enjoy your face recogniton model.
Step 3. (Optional) To change the color of text in the video, pick your color and change at line 16 in `synthetic.py` file



