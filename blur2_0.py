import cv2
import mediapipe as mp
import numpy as np

current_image = 'test1.png'

mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

BG_COLOR = (255, 255, 255)
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=1) as selfie_segmentation:
  image = cv2.imread(current_image)
  image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
  image.flags.writeable = False
  results = selfie_segmentation.process(image)
  image.flags.writeable = True
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

  condition = np.stack(
    (results.segmentation_mask,) * 3, axis=-1) > 0.1

  background = np.zeros(image.shape, dtype=np.uint8)
  background[:] = BG_COLOR
  
  output_image = np.where(condition, image, background)

  cv2.imshow('MediaPipe Selfie Segmentation', output_image)
  cv2.waitKey(0)
    

    