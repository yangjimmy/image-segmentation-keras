from keras_segmentation.models.unet import vgg_unet
import time
import cv2

model = vgg_unet(n_classes=51,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "tmp/vgg_unet_1" , epochs=25
)

start = time.time()
out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07965.png",
    out_fname="tmp/out.png"
)
print("time elapsed: {}".format(time.time()-start))

import matplotlib.pyplot as plt
plt.figure()
plt.imshow(out)
plt.figure()
plt.imshow(cv2.imread("dataset1/annotations_prepped_test/0016E5_07965.png")/51.0)
plt.show()

start = time.time()
out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/0016E5_07961.png",
    out_fname="tmp/out2.png"
)
print("time elapsed: {}".format(time.time()-start))


# evaluating the model 
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )
