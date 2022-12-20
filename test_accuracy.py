import tensorflow as tf
from tensorflow.python.keras.models import load_model
import numpy as np
import pathlib

# Defines the structure of the model, complies and trains it. Then saves it as "my_mdodel.json"
data_dir = "Training_data"
data_dir = pathlib.Path(data_dir)

model = load_model("D:/garima/tiet study material/prism/model/my_model.h5")
model.summary()


val_dataset = tf.keras.preprocessing.image_dataset_from_directory(data_dir,labels="inferred", validation_split= 0.2, subset= "validation", seed=123,  label_mode='categorical', color_mode='grayscale')


predictions = np.array([])
labels =  np.array([])
for x, y in val_dataset:
  predictions = np.concatenate([predictions, model.predict_classes(x)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])


print(predictions,  " " , labels)
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("Accuracy is " , accuracy_score(labels, predictions)*100)
print("Classification report is \n" , classification_report(labels, predictions))
