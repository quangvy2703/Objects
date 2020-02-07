python main.py   --input_video videos/videos.mp4 \
		 --output_video videos/results/video_ob.mp4 \
		 --train_dir datasets/train \
		 --test_dir datasets/test  \
		 --use_face_detection True \
		 --use_gender_prediction True \
		 --use_face_recognition True \
		 --use_scenes_change_count True \
		 --use_age_prediction True \
		 --emb_size 512 \
		 --n_classes 4 \
		 --gpuid 0 \
		 --use_emotion_prediction True \
#                 --use_objects_detection True \

