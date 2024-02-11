import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import warnings
import cv2
from object_detection.utils import visualization_utils as viz_utils
from object_detection.utils import label_map_util
warnings.filterwarnings('ignore')

category_index=label_map_util.create_category_index_from_labelmap("label_map.pbtxt")
# Function to load the saved model
def load_model(model_path):
    return tf.saved_model.load(model_path)

# Function to preprocess image
def preprocess_image(image):
    return tf.convert_to_tensor(image)[tf.newaxis, ...]

# Function to perform inference
def infer(model, image):
    input_tensor = preprocess_image(image)
    detections = model(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    return detections

# Load the model
model = load_model('saved_model/')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform inference
    detections = infer(model, frame)

    # Visualize detections on the frame
    viz_utils.visualize_boxes_and_labels_on_image_array(
        frame,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=1,
        min_score_thresh=0.5,  # Adjust as needed
        agnostic_mode=False)

    # Display the frame with detections
    cv2.imshow('Real-time Object Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
