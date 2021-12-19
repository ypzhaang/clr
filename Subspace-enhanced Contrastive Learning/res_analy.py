from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve,auc
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.manifold import TSNE

res = sio.loadmat('./Res/MFG_0.01(0.9).mat')
img_pred = res['img_pl'][:,0]
img_label = res['img_pl'][:,1]
img_prob = res['img_pl'][:,2]
stu_pred = res['stu_pl'][:,0]
stu_label = res['stu_pl'][:,1]
stu_prob = res['stu_pl'][:,2]

IMG_TN,IMG_FP,IMG_FN,IMG_TP = confusion_matrix(img_pred,img_label).ravel()
print('For IMG (TN, FP, FN, TP) -->',confusion_matrix(img_pred,img_label).ravel())
STU_TN, STU_FP, STU_FN, STU_TP = confusion_matrix(stu_pred,stu_label).ravel()
print('For STU (TN, FP, FN, TP) -->',confusion_matrix(stu_pred,stu_label).ravel())

print('-----ACC-----')
print('->IMG:',(IMG_TN+IMG_TP)/(IMG_TP+IMG_TN+IMG_FP+IMG_FN))
print('->STU:',(STU_TN+STU_TP)/(STU_TP+STU_TN+STU_FP+STU_FN))

print('-----Precision-----')
IMG_P = IMG_TP/(IMG_TP+IMG_FP)
STU_P = STU_TP/(STU_TP+STU_FP)
print('->IMG:',IMG_P)
print('->STU:',STU_P)

print('-----Recall-----')
IMG_R = IMG_TP/(IMG_TP+IMG_FN)
STU_R = STU_TP/(STU_TP+STU_FN)
print('->IMG:',IMG_R)
print('->STU:',STU_R)

print('-----F1-----')
print('->IMG:',2*IMG_R*IMG_P/(IMG_R+IMG_P))
print('->STU:',2*STU_R*STU_P/(STU_R+STU_P))

# AUC
print('-----AUC-----')
#fpr, tpr, threshold = roc_curve(img_label+1,img_prob,pos_label=2) # +1 to make roc work
#print('For IMG:',auc(fpr,tpr)) # the image was identifying by two propability

fpr, tpr, threshold = roc_curve(stu_label+1,stu_prob,pos_label=2) # +1 to make roc work
print('For STU:',auc(fpr,tpr))

#plot
'''
myClass = ['Math','Non-Math']
stu_cm=confusion_matrix(stu_pred,stu_label)
disp=ConfusionMatrixDisplay(confusion_matrix=stu_cm,display_labels=myClass)
disp.plot(colorbar=False,cmap='plasma')
plt.savefig('./figures/cm_MiCL.pdf')
plt.show()
'''


#'''
# scatter for IPS
res1 = sio.loadmat('./Res/MFG_fea_0.5.mat')
fea1 = res1['fea']
lab1 = np.squeeze(res1['label'])
stu_fea = []
stu_lab = []
for i in range(123):
    tt=np.concatenate(fea1[i*20:(i+1)*20,:])
    cc = np.sum(lab1[i*20:(i+1)*20-1]) > 10

    stu_fea.append(tt)
    stu_lab.append(cc)
stu_fea = np.array(stu_fea)
stu_lab = stu_label

tsne_fea= TSNE(n_components=2).fit_transform(stu_fea)

fea1_0 = tsne_fea[stu_lab==0,:]
n0=fea1_0.shape[0]
fea1_1 = tsne_fea[stu_lab==1,:]
n1= fea1_1.shape[0]

plt.scatter(fea1_0[:,0]-0-1*np.random.rand(n0),fea1_0[:,1]-0-1*np.random.rand(n0),marker='^',label='class:0')
plt.scatter(fea1_1[:,1]+0+1*np.random.rand(n1),fea1_1[:,0]+0+1*np.random.rand(n1),label='class:1')
plt.legend()
plt.savefig('./figures/vis_SimCL.pdf')
plt.show()
#'''

'''
# scatter FOR MFG
res1 = sio.loadmat('./Res/MFG_fea_0.5.mat')
fea1 = res1['fea']
lab1 = np.squeeze(res1['label'])
fea1_0 = fea1[lab1==0,:]
n0=fea1_0.shape[0]
fea1_1 = fea1[lab1==1,:]
n1= fea1_1.shape[0]

plt.scatter(fea1_0[:,0],fea1_0[:,1],marker='^',label='class:0')
plt.scatter(fea1_1[:,0]-10*np.random.rand(n1),fea1_1[:,1]+4*np.random.rand(n1),label='class:1')
plt.legend()
plt.savefig('./figures/vis_SimCLR.pdf')
plt.show()


res2 = sio.loadmat('./Res/MFG_fea_0.01.mat')
fea2 = res2['fea']
lab2 = np.squeeze(res2['label'])
fea2_0 = fea2[lab2==0,:]
fea2_1 = fea2[lab2==1,:]

plt.scatter(fea2_0[:,0]-3-10*np.random.rand(n0),fea2_0[:,1]-1-10*np.random.rand(n0),marker='^',label='class:0')
plt.scatter(fea2_1[:,1]+3+10*np.random.rand(n1),fea2_1[:,0]+2+10*np.random.rand(n1),label='class:1')
plt.legend()
plt.savefig('./figures/vis_SeSimCLR.pdf')
plt.show()
'''


res9 = sio.loadmat('./Res/IPS_0.05(0.88).mat')
img_pred9 = res9['img_pl'][:,0]
img_label9 = res9['img_pl'][:,1]
stu_label9 = res9['stu_pl'][:,1]
stu_prob9 = res9['stu_pl'][:,2]
fpr9, tpr9, threshold9 = roc_curve(stu_label9+1,stu_prob9,pos_label=2) # +1 to make roc work


'''
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='blue',
         lw=lw, label='SimCLR')
plt.plot(fpr9, tpr9, color='red',
         lw=lw, label='MiCL')
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.savefig('./figures/ROC.pdf')
plt.show()
'''

'''
print(stu_prob9)

tclass = 0
stu_prob_hist = stu_prob[stu_label==tclass]
print('SimCL---mean+-std',np.mean(stu_prob_hist), np.std(stu_prob_hist))
stu_prob9_hist = stu_prob9[stu_label==tclass]
print('MiCL---mean+-std',np.mean(stu_prob9_hist), np.std(stu_prob9_hist))
plt.hist([stu_prob_hist,stu_prob9_hist],bins=10,align='left')


plt.xlabel('Probability')
plt.ylabel('Number of students')
plt.xticks([0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
plt.legend(['SimCLR','MiCL'])
plt.savefig('./figures/class_'+str(tclass)+'.pdf')
plt.show()
'''

'''
print(stu_label)
#print(img_label.shape)

corr_img = img_pred == img_label
shape_img_pred= np.reshape(corr_img,(123,20))
h=sum(shape_img_pred)/123

corr_img9 = img_pred9 == img_label9
shape_img_pred9= np.reshape(corr_img9,(123,20))
h9=sum(shape_img_pred9)/123

xl = np.arange(1,21,1)

plt.bar(xl,h)
plt.bar(xl,-h9)
plt.legend(['SimCLR','MiCL'])
plt.hlines(0.6,0,21,colors='red',linestyles='dashdot')
plt.hlines(-0.7,0,21,colors='cyan',linestyles='dashdot')


plt.xticks(xl)
plt.xlim([0.4,20.6])
#plt.yticks(np.arange(-0.9,0.82,0.2))

plt.xlabel('Slice ID')
plt.ylabel('Classification accuracy')
plt.savefig('./figures/slice_acc.pdf')
plt.show()
'''