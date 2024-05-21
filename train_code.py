from keras_segmentation.models.unet import vgg_unet
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


model = vgg_unet(n_classes=5,  input_height=352, input_width=384)

model.train(
    train_images =  "dataset2\\images\\",
    train_annotations = "dataset2\\labels\\",
    checkpoints_path = "tmp\\vgg_unet" , epochs=1
)

# save model
model_json = model.to_json()

try:
    os.makedirs('models')
except:
    pass

fp = open('models/modelP_unet_vgg_1_epoch.json','w')
fp.write(model_json)
model.save_weights('models/modelW_unet_vgg_1_epoch.h5')

# TODO: reload model from saved weights here
# model2 = load_json_model('models/modelP_unet_vgg_1_epoch.json')
model2 = vgg_unet(n_classes=5,  input_height=352, input_width=384)


model2.load_weights('models/modelW_unet_vgg_1_epoch.h5')

# try on test data
start = time.time()
out = model2.predict_segmentation(
    inp="dataset2\\test_images\\scan27_1_VSCAN_0027_190.png",
    out_fname="tmp\\out_unet_epochs_1.png"
)
print(out.shape)
# prrint()
print("time elapsed: {}".format(time.time()-start))

plt.figure()
plt.imshow(out)
plt.figure()
plt.imshow(cv2.imread("dataset2\\test_images\\scan27_1_VSCAN_0027_190.png"))
plt.show()

start = time.time()
out = model2.predict_segmentation(
    inp="dataset2\\test_images\\scan27_1_VSCAN_0027_190.png",
    out_fname="tmp\\out2.png"
)
print("time elapsed: {}".format(time.time()-start))
print()


# evaluating the model 
# print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
