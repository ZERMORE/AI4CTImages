#coding=utf8

import sys
import os, glob
import matplotlib.pyplot as plt

def prepare_aapm_dicom_txt(filename, patient_list):

    root_dir = 'E://project_files/ctimages/task1/AAPM/'

    for patient in patient_list:

        ldct_img_dir = root_dir + patient + '/quarter_1mm/'
        ldct_img_list = os.listdir(ldct_img_dir)
        ldct_img_list.sort(key=lambda x: int(x[-13: -4]))

        rdct_img_dir = root_dir + patient + '/full_1mm/'
        rdct_img_list = os.listdir(rdct_img_dir)
        rdct_img_list.sort(key=lambda x: int(x[-13: -4]))

        for index in range(len(ldct_img_list)):

            ldct_img_file = ldct_img_dir + ldct_img_list[index]
            rdct_img_file = rdct_img_dir + rdct_img_list[index]

            paired_file = ldct_img_file + ' ' + rdct_img_file

            with open(filename, 'a') as f:
                f.write(paired_file + '\n')
            f.close()

if __name__ == "__main__":

    train_txt = './train_img.txt'
    valid_txt = './valid_img.txt'
    # test_txt = './test_img.txt'
    
    train_patient_list = ['L109', 'L291', 'L506', 'L192', 'L286', 'L143']
    valid_patient_list = ['L096', 'L333']
    # test_patient_list = ['L310', 'L067']
    
    if os.path.exists(train_txt) == False:
        prepare_aapm_dicom_txt(train_txt, train_patient_list)
    
    if os.path.exists(valid_txt) == False:
        prepare_aapm_dicom_txt(valid_txt, valid_patient_list)