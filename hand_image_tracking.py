import cv2
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For static images:
IMAGE_FILES = ['/res/ok1.jpg']
basePath = os.getcwd()
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5) as hands:
  for idx, file in enumerate(IMAGE_FILES):
    # Read an image, flip it around y-axis for correct handedness output (see
    # above).
    image = cv2.flip(cv2.imread(basePath+ file), 1)
    # Convert the BGR image to RGB before processing.
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Print handedness and draw hand landmarks on the image.
    print('Handedness:', results.multi_handedness)
    if not results.multi_hand_landmarks:
      continue
    image_height, image_width, _ = image.shape

    print('image width: ', image_width)
    print('image height: ', image_height)
    annotated_image = image.copy()

    print("=======Hand landmarks========")
    for hand_landmarks in results.multi_hand_landmarks:
      # print hand landmarks
      for id, landmark in enumerate(hand_landmarks.landmark):
        print(id,': ',landmark.x,'; ', landmark.y, '; ', landmark.z)

      mp_drawing.draw_landmarks(
          annotated_image,
          hand_landmarks,
          mp_hands.HAND_CONNECTIONS,
          mp_drawing_styles.get_default_hand_landmarks_style(),
          mp_drawing_styles.get_default_hand_connections_style())

    cv2.imwrite(basePath+ '/res/annotated_' + file[file.rfind('/')+1:file.rfind('.')] + '.png', cv2.flip(annotated_image, 1))

    # # Draw hand landmarks 
    # for hand_landmarks in results.multi_hand_world_landmarks: 
      # mp_drawing.plot_landmarks(
      #   hand_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)

    if not results.multi_hand_world_landmarks:
      continue

    print("=======Hand world landmarks========")
    for hand_world_landmarks in results.multi_hand_world_landmarks:
      # Print hand world landmarks
      for id, landmark in enumerate(hand_world_landmarks.landmark):
        print(id,': ',landmark.x,'; ', landmark.y, '; ', landmark.z)
      # Draw hand world landmarks.
      mp_drawing.plot_landmarks(
        hand_world_landmarks, mp_hands.HAND_CONNECTIONS, azimuth=5)
