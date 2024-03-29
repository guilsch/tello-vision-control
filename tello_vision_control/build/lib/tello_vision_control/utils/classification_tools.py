import tensorflow as tf
import numpy as np
import csv
import os
import tello_vision_control

package_dir = os.path.dirname(tello_vision_control.__file__)


def get_label_id_from_keyboard(key):
    """ Transform key returned by keyboard to number pressed

    Args:
        key (int): key

    Returns:
        _type_: number from 0 to 9
    """
    label_id = None
    if 48 <= key <= 57:  # 0 ~ 9
        label_id = key - 48
    return label_id

def write_new_data(label_id, landmark_list, csv_path='data/keypoint.csv'):
    """ Add data entry to the data for training
    """
    
    with open(csv_path, 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([label_id, *landmark_list])
        
    return

def create_landmarks_classifier(num_class):
    """ Create model to be trained

    Args:
        num_class (int): number of possible outputs (ex: 2 for 'open' and 'close')

    Returns:
        _type_: model
    """
    model = tf.keras.models.Sequential([
    tf.keras.layers.Input((21 * 2, )),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(20, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(num_class, activation='softmax')
    ])
    
    return model


class KeyPointClassifierLoader(object):
    """ TF Lite interpreter corresponding to the classifier used in inference, where the 
        model is specified in the model_path arg.
    """
    
    def __init__(
        self,
        model_path=None,
        num_threads=1,
    ):
        if model_path is None:
            model_path = os.path.join(package_dir, 'data/gesture_classifier', 'keypoint_classifier.tflite')
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(self, landmark_list,):
        """ Inference, returns label id
        """
        
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']
        result = self.interpreter.get_tensor(output_details_tensor_index)
        result_index = np.argmax(np.squeeze(result))

        return result_index
    
def save_classifier_to_TFLite(classifier, tflite_save_path = 'data/keypoint_classifier.tflite'):
    """ Save the TF classifier to TFLite in the specified file

    Args:
        classifier (_type_): TF classifier (not TFLite)
        tflite_save_path (str, optional): file where the TFLite model will be saved
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, 'wb').write(tflite_quantized_model)