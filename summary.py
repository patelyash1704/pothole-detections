from tensorflow.keras.models import load_model
model=load_model("resnet50_2.h5")
model.summary()