import numpy as np
import os
import glob
import nibabel as nib
outdir = '../Train'
if not os.path.exists(outdir):
    os.makedirs(outdir)

files = glob.glob("/home/psoni/Desktop/project/Train/*.nii.gz")
for filepath in files:
    file = np.array(nib.load(filepath).dataobj)
    print('  Data shape is ' + str(file.shape) + ' .')
    for i in range(0, 200):
        x = int(np.floor((file.shape[0] - 64) * np.random.rand(1))[0])
        y = int(np.floor((file.shape[1] - 64) * np.random.rand(1))[0])
        z = int(np.floor((file.shape[2] - 64) * np.random.rand(1))[0])
        file_aug = file[x:x+64, y:y+64, z:z+64, :]
        filename_ = filepath.split('/')[-1].split('.')[0]
        filename_ = filename_ + '_' + str(i) + '.nii.gz'
        filename = os.path.join(outdir, filename_)
        new_image = nib.Nifti1Image(file_aug, affine=np.eye(4))
        nib.save(new_image, "filename")
        print(str(i))
    print('All sliced files of ' + filepath + ' are saved.')
