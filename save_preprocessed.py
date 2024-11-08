import cv2
from tqdm import tqdm
import numpy as np
import os

def preprocess(scan_dirs, parent_dir, n_classes = 5, img_format='png'):
    # X = []
    # Y = []
    for scan_dir in scan_dirs:
        image_idx = 0

        img_dir = "{}\{}".format(parent_dir, scan_dir)
        label_dir = "{}\{}\LabelingProject\GroundTruthProject\PixelLabelData".format(parent_dir, scan_dir)

        img_files = next(os.walk(img_dir))[2]
        msk_files = next(os.walk(label_dir))[2]

        img_files.sort()
        msk_files.sort()

        print(len(img_files))
        print(len(msk_files))

        for img_fl in tqdm(img_files):    
            if(img_fl.split('.')[-1]==img_format):
                image_idx += 1

                img = cv2.imread('{}\{}'.format(img_dir,img_fl), cv2.IMREAD_GRAYSCALE)

        #         print('E:\YKL\Thorlabs VSCAN Labeling\scan30\{}'.format(img_fl))
                resized_img = cv2.equalizeHist(np.clip(cv2.resize(img,(384, 352), interpolation = cv2.INTER_NEAREST),0,255))
                resized_denoised_img = cv2.fastNlMeansDenoising(resized_img,10,7,21)
                resized_denoised_img = np.stack((resized_denoised_img,resized_denoised_img,resized_denoised_img),axis=2)

                # write to file
                cv2.imwrite('dataset3\\images\\{}_{}_{}'.format(scan_dir,image_idx,img_fl),resized_denoised_img)
                
                # X.append(resized_denoised_img)

                msk = cv2.imread('{}\\Label_{}_{}'.format(label_dir,image_idx,img_fl.split('.')[0]+'.png'), cv2.IMREAD_GRAYSCALE)
                resized_msk = np.clip(cv2.resize(msk,(384, 352), interpolation = cv2.INTER_NEAREST),0,n_classes-1)
                resized_msk = np.stack((resized_msk,np.zeros((352,384)),np.zeros((352,384))),axis=2)
                # print(resized_msk[:,:,2])
                # if np.max(resized_msk[:,:,1:3]) != 0 or np.min(resized_msk[:,:,1:3]) != 0:
                #     raise ValueError
                cv2.imwrite('dataset3\\labels\\{}_{}_{}'.format(scan_dir,image_idx,img_fl),resized_msk)

                # resized_msk = np.clip(cv2.resize(msk,(400, 352), interpolation = cv2.INTER_NEAREST),0,4)

                # additional post processing for one hot encoding mask
                # resized_msk[resized_msk==0] = 6;
                # resized_msk_one_hot = np.zeros((resized_msk.shape[0], resized_msk.shape[1], n_classes))
                # for i, unique_value in enumerate(np.unique(resized_msk)):
                #     resized_msk_one_hot[:, :, i][resized_msk == unique_value] = 1

                # Y.append(resized_msk_one_hot)

if __name__ == '__main__':
    parent_dir = "E:\YKL\Thorlabs VSCAN Labeling"
    preprocess(['scan30','scan31','scan32'])

    # preprocess(['scan27'])