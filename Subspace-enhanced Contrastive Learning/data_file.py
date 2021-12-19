import nibabel as nib
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import pyreadstat
import random
import cv2 as cv
from PIL import Image

# read Data.sav
df, meta=pyreadstat.read_sav('./MRS_Data/Data.sav')

# read images
mrs_path='/Users/zhyp/Documents/PycharmProjects/proj-SCL/MRS_Data/PN21_MRS/central.xnat.org'
roi_path='./MRS_Data/PN21_preprocessed/central.xnat.org/'

#123 =
n_student = 123
#for IPS
#sav_nonmath = './path/nonmath/' #labe is 1
#sav_math = './path/math/' # label is 0
#for MFG
sav_nonmath = './path_mfg/nonmath/' #labe is 1
sav_math = './path_mfg/math/' # label is 0

# for val option{path_mfg_val, path_ips_val}
sav_tr_nonmath = './path_mfg_val/tr/nonmath'
sav_tr_math = './path_mfg_val/tr/math'
sav_te_nonmath = './path_mfg_val/te/nonmath'
sav_te_math = './path_mfg_val/te/math'

isval = True
random.seed(1)
IDX_tr = np.array(random.sample(range(0,n_student),43))

# get max() and min()
max_pix = 0
min_pix = pow(2,16)

# read roi
g_roi = os.walk(roi_path)

for _, dir_list, _ in g_roi:
    conut_s = 0
    for dir_name in sorted(dir_list):
        # read roi
        stu_path = os.path.join(roi_path,dir_name)+'/preprocessed'
        mIPS_path = stu_path+'/mIPS.nii'
        mMFG_path = stu_path+'/mMFG.nii'
        mIPS_img = nib.load(mIPS_path).get_fdata()
        mIPS_crop = np.nonzero(mIPS_img)
        mMFG_img = nib.load(mMFG_path).get_fdata()
        mMFG_crop = np.nonzero(mMFG_img)
        #read TI MRS max = 1301; min =0; for all MRS
        stu_t1_mrs_path = os.path.join(mrs_path,dir_name) + '/2/NIFTI/s' + dir_name[0:-4] +'.nii'
        stu_t1_mrs_img = nib.load(stu_t1_mrs_path).get_fdata()

        # plot
        '''
        if 1:
            id_tem = mIPS_crop[2][0]+1
            plt_img = stu_t1_mrs_img[id_tem,:,:]
            xmin, xmax = mIPS_crop[2][0],mIPS_crop[2][-1]+1
            ymin, ymax = mIPS_crop[1][0],mIPS_crop[1][-1]+1
        else:
            id_tem = mMFG_crop[0][0]-2
            plt_img = stu_t1_mrs_img[id_tem,:,:]
            xmin, xmax = mMFG_crop[0][0],mMFG_crop[0][-1]+1
            ymin, ymax = mMFG_crop[1][0],mMFG_crop[1][-1]+1

        figure, ax = plt.subplots(1)
        rect = patches.Rectangle((xmin,ymin),20,20, edgecolor='c', facecolor="none",linestyle='-.',linewidth=2.5)
        ax.imshow(plt_img,cmap='gray',origin='lower')#,origin='lower'
        ax.axis("off")
        ax.add_patch(rect)
        plt.show()
        figure.savefig('./figures/IPS_segittal.pdf')
        plt.close(figure)
        '''

        # crop
        #IPS: max = 334; min = 7
        #MFG: max = 321; min = 15
        mIPS = stu_t1_mrs_img[mIPS_crop[0][0]:mIPS_crop[0][-1]+1,mIPS_crop[1][0]:mIPS_crop[1][-1]+1,mIPS_crop[2][0]:mIPS_crop[2][-1]+1]
        mIPS = (mIPS-7)/(334-7)
        mMFG = stu_t1_mrs_img[mMFG_crop[0][0]:mMFG_crop[0][-1]+1,mMFG_crop[1][0]:mMFG_crop[1][-1]+1,mMFG_crop[2][0]:mMFG_crop[2][-1]+1]
        mMFG = (mMFG-15)/(321-15)
        if mMFG.max() > max_pix:
            max_pix = mMFG.max()
        if mMFG.min() < min_pix:
            min_pix = mMFG.min()

        stu_id = int(dir_name[12:15])
        gnd=df.lackMaths[int(np.where(df.scanID == stu_id)[0])]
        #save
        if isval:
            if sum(IDX_tr == conut_s) == 0:
                sav_nonmath = sav_tr_nonmath
                sav_math = sav_tr_math
            else:
                sav_nonmath = sav_te_nonmath
                sav_math = sav_te_math
        if gnd == 1: # label is 1, is lack math
            sav_img = sav_nonmath
        else:
            sav_img = sav_math

        for i in range(20):
            # for IPS and MFG
            plt.imsave(os.path.join(sav_img,dir_name[12:15])+ '_' + str(i).zfill(2) +'.png', mIPS[i],cmap='gray')

        conut_s = conut_s+1
        print(mIPS.shape,mMFG.shape,conut_s,' **** ',dir_name,dir_name[12:15],gnd)

    break

print('max pixel value:',max_pix,'-- min pixel value',min_pix)
# save data
