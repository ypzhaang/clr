import h5py
import matplotlib.pyplot as plt

fData = h5py.File('./MRS_Data/ips-mfg.h5','r')
ips = fData['ips'][:]
mfg = fData['mfg'][:]
gnd = fData['gnd'][:]
name = fData['name'][:]
fData.close()

#################################################


#################################################
sav_nonmath='./path/nonmath/' #labe is 1
sav_math='./path/math/' # label is 0

pern=0
for i in range(1020+1440):
    pern = pern+1
    if gnd[i] == 1: #nonmath
        sav_path = sav_nonmath + str(int(name[i])).zfill(2) + '_' + str(pern).zfill(2) + '.png'
        plt.imsave(sav_path,ips[i],cmap = 'gray')
    else:
        sav_path = sav_math + str(int(name[i])).zfill(2) + '_' + str(pern).zfill(2) + '.png'
        plt.imsave(sav_path,ips[i],cmap = 'gray')
    print(i,':',sav_path)
    if pern >= 20:
        pern = 0