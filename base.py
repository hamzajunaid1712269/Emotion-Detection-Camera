from tensorflow.keras.models import model_from_json
import numpy as np
import tensorflow as tnf

class FEM(object):
    expressions_list = ["Angry","Disgust","Fear","Happy","Neutral","Sad","Surprise"]

    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_emotion(self, img):
        self.preds = self.loaded_model.predict(img)
        return FEM.expressions_list[np.argmax(self.preds)]
