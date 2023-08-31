from PIL import Image
import matplotlib 
import matplotlib.pyplot as plt
import os
import csv
import numpy as np
import tensorflow as tf
import numpy as np 
from six import BytesIO 
from PIL import Image
import tensorflow as tf  
from object_detection.utils import label_map_util 
from object_detection.utils import config_util 
from object_detection.utils import visualization_utils as viz_utils 
from object_detection.builders import model_builder 
import random


# initializing array for ground truth bounding boes and classes
ground_truth_boxes = []
ground_truth_classes = []

#initialize variable for test image 
test_image = None

#opens images from the order specified in a csv file (see README file)
def open_images(image_path):
  global test_image
  ordered_images = []
  with open(image_path, 'r') as file:
    csvreader = csv.reader(file)
    for row in csvreader: 

      ground_truth_boxes.append(tf.convert_to_tensor(np.array([[float(row[5])/float(row[2]), float(row[4])/float(row[1]), float(row[7])/float(row[2]), float(row[6])/float(row[1])]]), dtype=tf.float32))
      ground_truth_classes.append(tf.convert_to_tensor([[1]], dtype=tf.float32))

      if len(np.asarray(Image.open(f'images/{row[0]}')).shape) == 2:
         RGB_image = convert_grayscale_to_RGB(np.asarray(Image.open(f'images/{row[0]}')))
         ordered_images.append(RGB_image)

      else:
        test_image = np.asarray(Image.open(f'images/{row[0]}'))
        ordered_images.append(np.asarray(Image.open(f'images/{row[0]}')))
  
  return ordered_images

# converting any image that is grayscale to rgb format
def convert_grayscale_to_RGB(numpy_image):

  numpy_image = numpy_image[:, :, np.newaxis]
  numpy_image = numpy_image.tolist()

  for i in range(len(numpy_image)):
    for j in range(len(numpy_image[i])):
        for k in range(2):
          numpy_image[i][j].append(numpy_image[i][j][0])

  return np.array(numpy_image)

# load a pre-trained model that saved to your local directory 
def load_saved_model(model, path):

  trained_checkP = tf.train.Checkpoint(model = model)
  image, shapes = model.preprocess(tf.zeros([1, 640, 640, 3])) 
  prediction_dict = model.predict(image, shapes) 
  _ = model.postprocess(prediction_dict, shapes)
  trained_checkP.restore(path)

# feeding the model with the test data
def test_model(model, shapes, ground_truth_boxes, ground_truth_classes, ordered_images):
   
  test = []

  for i in range(len(ordered_images)):
    ordered_images[i] = tf.convert_to_tensor(np.expand_dims(ordered_images[i], 0), dtype=tf.float32)
    test.append(model.preprocess(ordered_images[i])[0])
    
  preprocessed_images_tf = tf.concat(test, axis=0)

  model.provide_groundtruth(groundtruth_boxes_list = ground_truth_boxes,
            groundtruth_classes_list = ground_truth_classes)

  prediction_dict = model.predict(preprocessed_images_tf, shapes)
  losses_dict = model.loss(prediction_dict, shapes)
  total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']

  return total_loss

#outside function to speed up training process (gets the negative gradient step of the function)
def get_model_train_step_function(model, optimizer, trainable_layers):

  @tf.function
  def train_step_fn(train_images):
    
    shapes = tf.constant([10 * [[320, 320, 3]]], dtype=tf.int32)

    with tf.GradientTape() as tape:
        prediction_dict = model.predict(train_images, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        gradients = tape.gradient(total_loss, trainable_layers)
        optimizer.apply_gradients(zip(gradients, trainable_layers))

    return total_loss
  
  return train_step_fn

# trains an already-trained model from the object-detection
def transfer_learning(model, ground_truth_boxes, ground_truth_classes, optimizer, ckpt_path, ordered_images):
  fake_box_predictor = tf.compat.v2.train.Checkpoint(
    _base_tower_layers_for_heads=detection_model._box_predictor._base_tower_layers_for_heads,
    #_prediction_heads=detection_model._box_predictor._prediction_heads,
    #    (i.e., the classification head that we *will not* restore)
    _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )

  fake_model = tf.compat.v2.train.Checkpoint(
            _feature_extractor=model._feature_extractor,
            _box_predictor=fake_box_predictor)
  ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
  ckpt.restore(ckpt_path).expect_partial()

  image, shapes = model.preprocess(tf.zeros([1, 640, 640, 3]))
  prediction_dict = model.predict(image, shapes)
  _ = model.postprocess(prediction_dict, shapes)

  test = []

  for i in range(len(ordered_images)):
    ordered_images[i] = tf.convert_to_tensor(np.expand_dims(ordered_images[i], 0), dtype=tf.float32)
    test.append(model.preprocess(ordered_images[i])[0])
    
  preprocessed_images_tf = tf.concat(test, axis=0)

  trainable_variables = model.trainable_variables

  trainable_layer_name = ["WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalBoxHead", "WeightSharedConvolutionalBoxPredictor/WeightSharedConvolutionalClassHead"]

  trainable_layers = []

  for layer in trainable_variables:
    if layer.name.startswith(trainable_layer_name[0]) or layer.name.startswith(trainable_layer_name[1]):
        trainable_layers.append(layer)

  model.provide_groundtruth(groundtruth_boxes_list = ground_truth_boxes,
            groundtruth_classes_list = ground_truth_classes)

  #optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)

  train_step_fn = get_model_train_step_function(model, optimizer, trainable_layers)

  for i in range(100):
    

    loss = train_step_fn(preprocessed_images_tf)

    if i % 10 == 0:
        print(loss, flush=True)

#if in eval_config it specificies keypoint edges, returns (star, end) of the keypoints. (Used for landmark detections)
def get_keypoint_tuples(eval_config):
   """Return a tuple list of keypoint edges from the eval config.
    Args:
     eval_config: an eval config containing the keypoint edges
    Returns:
     a list of edge tuples, each in the format (start, end)
   """
   tuple_list = []
   kp_list = eval_config.keypoint_edge
   for edge in kp_list:
     tuple_list.append((edge.start, edge.end))
   return tuple_list

#returns the dimensions and coordinates of the bounding boxes produced by the model
def get_model_detection_function(model):

    """Get a tf.function for detection."""
    @tf.function
    def detect_fn(image):
        """Detect objects in image."""
        image, shapes = model.preprocess(image)

        prediction_dict = model.predict(image, shapes)

        detections = model.postprocess(prediction_dict, shapes) 

        return detections, prediction_dict, tf.reshape(shapes, [-1])
    return detect_fn

#get the bounding boxes over the input image 
def get_predictions(model, category_index, input_image):

  #creating the detections object 
  detect_fn = get_model_detection_function(model)

  #getting all the classes for the model
  label_map_path = 'object_detection/data/mscoco_label_map.pbtxt'
  label_map = label_map_util.load_labelmap(label_map_path) 
  categories = label_map_util.convert_label_map_to_categories(
      label_map,
      max_num_classes=label_map_util.get_max_label_map_index(label_map),
      use_display_name=True) 

  #formatting all the classes
  #category_index = label_map_util.create_category_index(categories) 
  
  #converting the image into a tensor
  input_tensor = tf.convert_to_tensor(np.expand_dims(input_image, 0), dtype=tf.float32) 

  #getting the detections (bounding boxes)
  detections, predictions_dict, shapes = detect_fn(input_tensor)  

  label_id_offset = 1 
  image_np_with_detections = input_image.copy()  
  keypoints, keypoint_scores = None, None 

  #if there are keypoints that need to be detected, set them to a value. if not they will be set as None. 
  if 'detection_keypoints' in detections:
    keypoints = detections['detection_keypoints'][0].numpy()
    keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

  #overlaying the bounding over the image. This function groups all the bounding boxes in the same area as one
  viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'][0].numpy(),
        (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
        detections['detection_scores'][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=.50,
        agnostic_mode=False,
        keypoints=keypoints,
        keypoint_scores=keypoint_scores,
        keypoint_edges=get_keypoint_tuples(configs['eval_config']))

  #opening and showing the image
  img = Image.fromarray(image_np_with_detections, "RGB")
  img.show()

#getting the ordered images for training/testing
ordered_images = open_images("data/test_labels.csv")

#path of the .config file
pipeline_config = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/pipeline.config" 

#path of the pre-trained model's checkpoint
model_dir = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/checkpoint"

#getting the configuration of the model from the .config file
configs = config_util.get_configs_from_pipeline_file(pipeline_config) 

#getting the configuration of the model
model_config = configs['model']
model_config.ssd.num_classes = 1
model_config.ssd.freeze_batchnorm = True

#bulding the model based on the configuration
detection_model = model_builder.build(model_config=model_config, is_training=False)

#loading the pre-trained model that was saved in the local directory
load_saved_model(detection_model, "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_new_model/ckpt-1")

#outputing the image with bounding boxes
get_predictions(detection_model, category_index = { 1: {'id': 1, 'name': 'raccoon'}}, input_image = test_image)