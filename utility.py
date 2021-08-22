"""

    Created on 28/04/21 12:23 PM
    @author: Kartik Prabhu

"""
import glob
import os
import shutil
import fnmatch

from utils import interpolate_and_save
from VoxelInterpolation import convert
import matplotlib.pyplot as plt
import io

class Utility:

    # def __init__(self):

    def get_classes(self, root_path, model_folderName='models'):
        return [class_folder for class_folder in os.listdir(root_path+"/"+model_folderName+"/") if not class_folder.startswith('.')]


    def copy_files_to_destination(self,root_path,destination_path,model_folderName='models'):
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        destination_img_path = self.make_folder(destination_path +"/imgs")
        destination_masks_path = self.make_folder(destination_path +"/masks")
        destination_depth_path = self.make_folder(destination_path +"/depthmaps")
        destination_model_path = self.make_folder(destination_path +"/model")
        destination_normal_path = self.make_folder(destination_path +"/normal")

        for label in self.get_classes(root_path,model_folderName):
            destination_img_label_path = self.make_folder(os.path.join(destination_img_path,label))
            destination_masks_label_path = self.make_folder(os.path.join(destination_masks_path,label))
            destination_depth_label_path = self.make_folder(os.path.join(destination_depth_path,label))
            destination_model_label_path = self.make_folder(os.path.join(destination_model_path,label))
            destination_normal_label_path = self.make_folder(os.path.join(destination_normal_path,label))

            labelPath = os.path.join(root_path +model_folderName+"/", label)

            for folder in os.listdir(labelPath):
                if folder =='.DS_Store':
                    continue

                destination_folder_img_path = self.make_folder(os.path.join(destination_img_label_path,folder))
                destination_folder_mask_path = self.make_folder(os.path.join(destination_masks_label_path,folder))
                destination_folder_depth_path = self.make_folder(os.path.join(destination_depth_label_path,folder))
                destination_folder_model_path = self.make_folder(os.path.join(destination_model_label_path,folder))
                destination_folder_normal_path = self.make_folder(os.path.join(destination_normal_label_path,folder))

                # model_folder_path = os.path.join(labelPath, folder)
                # shutil.copyfile(os.path.join(model_folder_path,"model.obj"), os.path.join(destination_folder_model_path, "model.obj"))
                # shutil.copyfile(os.path.join(model_folder_path,"voxel.mat"), os.path.join(destination_folder_model_path, "voxel.mat"))

                # imgFolderPath  = os.path.join(labelPath, folder+'/img/')
                imgFolderPath  = os.path.join(labelPath, folder)

                print(imgFolderPath)
                fileCount = len(fnmatch.filter(os.listdir(imgFolderPath), '*.png'))
                print(fileCount/5)

                subscripts = ["_img","_normals","_id","_depth","_layer"]
                for i in range(0,int(fileCount/5)):
                    for sub in subscripts:
                        filename = str(i)+sub+".png"
                        print(filename)

                        try:
                            if filename.__contains__(str(i)):
                                if filename.__contains__("img"):
                                    # shutil.copyfile(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_img_path, str(i)+".png"))
                                    shutil.move(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_img_path, str(i)+".png"))
                            if filename.__contains__("depth"):
                                shutil.move(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_depth_path, str(i)+".png"))

                            if filename.__contains__("layer"):
                                shutil.move(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_mask_path, str(i)+".png"))

                            if filename.__contains__("normal"):
                                shutil.move(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_normal_path, str(i)+".png"))

                            if filename.__contains__("id"):
                                os.rename(os.path.join(imgFolderPath,filename), os.path.join(imgFolderPath, str(i)+".png"))
                        except:
                            print("error in "+ filename)

    def make_folder(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        return path

    def generate_voxels(self,root_path,model_foldername='models'):
        for label in self.get_classes(root_path,model_foldername):
            labelPath = os.path.join(root_path + model_foldername+"/", label)
            for folder in os.listdir(labelPath):
                if folder =='.DS_Store':
                    continue

                model_folder_path = os.path.join(labelPath, folder)

                voxel_path = os.path.join(model_folder_path,"voxel.mat")
                if not os.path.isfile(voxel_path) :
                    for fileName in os.listdir(model_folder_path):
                        if fileName.endswith(".mat"):
                            os.rename(os.path.join(model_folder_path,fileName), voxel_path)
                            break

                interpolate_and_save(voxel_path,model_folder_path,size=32, mode='nearest')
                interpolate_and_save(voxel_path,model_folder_path,size=64,mode='nearest')
                interpolate_and_save(voxel_path,model_folder_path,size=128,mode='nearest')

    def generate_voxels_2(self,root_path,outputPath,thresh,model_foldername='models'):

        self.make_folder(outputPath)
        for label in self.get_classes(root_path,model_foldername):
            if  label=='chair' or label=='bed' or label=='wardrobe'or label=='desk': #TODO: remove this check
                continue

            labelPath = os.path.join(root_path + model_foldername+"/", label)
            self.make_folder(os.path.join(outputPath, label))
            for folder in os.listdir(labelPath):
                if folder =='.DS_Store' :
                    continue

                model_folder_path = os.path.join(labelPath, folder)
                output_folder_path = os.path.join(os.path.join(outputPath, label),folder)
                self.make_folder(output_folder_path)
                voxel_path = os.path.join(model_folder_path,"voxel.mat")
                if not os.path.isfile(voxel_path) :
                    for fileName in os.listdir(model_folder_path):
                        if fileName.endswith(".mat"):
                            os.rename(os.path.join(model_folder_path,fileName), voxel_path)
                            break

                convert(voxel_path,output_folder_path,thresh, size=(32,32,32))
                convert(voxel_path,output_folder_path,thresh, size=(64,64,64))
                convert(voxel_path,output_folder_path,thresh, size=(128,128,128))

if __name__ == '__main__':
    root_path = '/Users/apple/OVGU/Thesis/s2r3dfree_v1_1'
    destination_path = '/Users/apple/OVGU/Thesis/s2r3dfree_v1_1_final/'

    root = "/Users/apple/OVGU/Thesis/SynthDataset1/"
    # root='/nfs1/kprabhu/SynthDataset1/'
    # root = '/Users/apple/OVGU/Thesis/Dataset/pix3d/'
    # root='/nfs1/kprabhu/pix3d/'

    #Copy file
    Utility().copy_files_to_destination(root_path=root_path,destination_path=destination_path,model_folderName="")
    # Utility().generate_voxels(root_path=root, model_foldername='model')

    # Utility().generate_voxels_2(root_path=root, outputPath=destination_path, thresh=0.1,model_foldername='models')