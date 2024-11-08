from keras_segmentation.models.unet import mobilenet_unet
import time
import cv2
import matplotlib.pyplot as plt
import os
from keras.models import Model, model_from_json

def load_json_model(model_name):
    json_file = open(model_name, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    return loaded_model


model = mobilenet_unet(n_classes=2,  input_height=32, input_width=64)

model.train(
    train_images =  "dataset3\\images\\",
    train_annotations = "dataset3\\labels\\",
    checkpoints_path = "tmp\\unet_dental", epochs=1
)

# save model
model_json = model.to_json()

try:
    os.makedirs('models')
except:
    pass

fp = open('models/modelP_dental.json','w')
fp.write(model_json)
model.save_weights('models/modelW_dental.h5')

# TODO: reload model from saved weights here
# model2 = load_json_model('models/modelP_unet_vgg_1_epoch.json')
model2 = mobilenet_unet(n_classes=2,  input_height=32, input_width=64)

model2.load_weights('models/modelW_dental.h5')

# try on test data
start = time.time()
out = model2.predict_segmentation(
    inp="dataset3\\test_images\\08.png",
    out_fname="tmp\\out_dental_epochs_1.png"
)
print(out.shape)
# prrint()
print("time elapsed: {}".format(time.time()-start))

plt.figure()
plt.imshow(out)
plt.figure()
plt.imshow(cv2.imread("dataset3\\test_images\\08.png"))
plt.show()

start = time.time()
out = model2.predict_segmentation(
    inp="dataset3\\test_images\\08.png",
    out_fname="tmp\\out2.png"
)
print("time elapsed: {}".format(time.time()-start))
print()


# evaluating the model 
# print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
