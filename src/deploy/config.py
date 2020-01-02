import cv2

THRESHOLD_DISTANCE = -30
THRESHOLD_SIMILARITY = 0.1
COUNTDOWN = 3
MAX_IMAGES = 300

ELLIPSE_WIDTH = 170
ELLIPSE_HEIGHT = 220

RED = (255, 0, 0)
BLACK = (0, 0, 0)
PINK = (255, 195, 255)
BLUE = (181, 212, 66)
TEAL = (255, 255, 0)
ORANGE = (122, 170, 255)
GREEN = (0, 255, 0)


FONT = cv2.FONT_HERSHEY_SIMPLEX
PLOT_CHART = False


TRAIN_DIR = '/media/vy/DATA/projects/face/insightface_clone/deploy/images'
TEST_DIR = '/media/vy/DATA/projects/face/insightface/sample_images/All_Age_Faces_Dataset/original_images'
NEGATIVE_DIR = '/media/vy/DATA/data/original images'

PIVOT_DIR = '/media/vy/DATA/projects/face/insightface_clone/deploy/images/results/'