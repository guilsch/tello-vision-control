import tensorflow as tf
import numpy as np


def save_classifier_to_TFLite(classifier, tflite_save_path = 'data/keypoint_classifier.tflite'):
    """ Save the TF classifier to TFLite in the specified adress

    Args:
        classifier (_type_): TF classifier (not TFLite)
        tflite_save_path (str, optional): adress where the TFLite model will be saved
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_quantized_model = converter.convert()

    open(tflite_save_path, 'wb').write(tflite_quantized_model)


class KeyPointClassifierLoader(object):
    """ Object corresponding to the TF Lite classifier, where the 
        model is specified in the model_path arg
    """
    
    def __init__(
        self,
        model_path='data/keypoint_classifier.tflite',
        num_threads=1,
    ):
        self.interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)

        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        input_details_tensor_index = self.input_details[0]['index']
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        self.interpreter.invoke()

        output_details_tensor_index = self.output_details[0]['index']

        result = self.interpreter.get_tensor(output_details_tensor_index)

        result_index = np.argmax(np.squeeze(result))

        return result_index