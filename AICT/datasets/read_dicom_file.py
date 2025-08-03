import torch
import pydicom
from torch.utils.data import Dataset
import numpy as np

class dicom_reader(torch.utils.data.Dataset):
    def __init__(self, paired_img_txt):
        super(dicom_reader, self).__init__()

        self.paired_files = open(paired_img_txt).readlines()

    def __len__(self):
        return len(self.paired_files)

    def to_tensor(self, data):
        if data.ndim == 2:
            data = data[np.newaxis, ...]
        elif data.ndim == 3:
            data = data.transpose(2, 0, 1)

        data = torch.FloatTensor(data)

        return data

    def to_numpy(self, data):
        data = data.detach().cpu().numpy()
        data = data.squeeze()
        if data.ndim == 3: data = data.transpose(1, 2, 0)
        return data

    def get(self, paired_files):

        ldctImg = pydicom.dcmread(paired_files[0: paired_files.index('.IMA ') + 4])
        ldctImg = ldctImg.pixel_array
        ldctImg[ldctImg < 0] = 0

        rdctImg = pydicom.dcmread(paired_files[paired_files.index('.IMA ')+5:-1])
        rdctImg = rdctImg.pixel_array
        rdctImg[rdctImg < 0] = 0

        ldctImg = np.array(ldctImg).astype(np.float32)
        rdctImg = np.array(rdctImg).astype(np.float32)

        ldctImg = self.to_tensor(ldctImg)
        rdctImg = self.to_tensor(rdctImg)

        return {"ldct": ldctImg, "rdct": rdctImg}

    def __getitem__(self, index):
        paired_files = self.paired_files[index]
        return self.get(paired_files)

# if __name__ == "__main__":
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     dataset = MedicalImage()
#     data = dataset[100]
#
#     ldctImg = dataset.to_numpy(data["degeneration"]).astype(np.uint16)
#     rdctImg = dataset.to_numpy(data["reference"]).astype(np.uint16)
#
#     plt.imshow(rdctImg-ldctImg, cmap=plt.cm.gray)
#     plt.show()