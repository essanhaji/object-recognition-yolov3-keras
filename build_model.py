from tools import make_yolov3_model
from weight_reader import WeightReader

# define the model
model = make_yolov3_model()

# load the model weights
weight_reader = WeightReader('yolov3.weights')

# set the model weights into the model
weight_reader.load_weights(model)

# save the model to file
model.save('yolo3_keras.h5')