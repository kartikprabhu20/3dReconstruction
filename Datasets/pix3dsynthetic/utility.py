"""

    Created on 28/04/21 12:23 PM
    @author: Kartik Prabhu

"""
import os
import shutil

from utils import interpolate_and_save


class Utility:

    # def __init__(self):

    def get_classes(self, root_path, model_folderName='models'):
        return [class_folder for class_folder in os.listdir(root_path+"/"+model_folderName+"/") if not class_folder.startswith('.')]


    def copy_files_to_destination(self,root_path,destination_path,model_folderName='models'):
        if not os.path.exists(destination_path):
            os.mkdir(destination_path)

        destination_img_path = destination_path +"/imgs"
        destination_masks_path = destination_path +"/masks"
        destination_depth_path = destination_path +"/depthmaps"
        destination_model_path = destination_path +"/model"

        if not os.path.exists(destination_img_path):
            os.mkdir(destination_img_path)
        if not os.path.exists(destination_masks_path):
            os.mkdir(destination_masks_path)
        if not os.path.exists(destination_depth_path):
            os.mkdir(destination_depth_path)
        if not os.path.exists(destination_model_path):
            os.mkdir(destination_model_path)


        for label in self.get_classes(root_path,model_folderName):
            destination_img_label_path = os.path.join(destination_img_path,label)
            destination_masks_label_path = os.path.join(destination_masks_path,label)
            destination_depth_label_path = os.path.join(destination_depth_path,label)
            destination_model_label_path = os.path.join(destination_model_path,label)

            if not os.path.exists(destination_img_label_path):
                os.mkdir(destination_img_label_path)
            if not os.path.exists(destination_masks_label_path):
                os.mkdir(destination_masks_label_path)
            if not os.path.exists(destination_depth_label_path):
                os.mkdir(destination_depth_label_path)
            if not os.path.exists(destination_model_label_path):
                os.mkdir(destination_model_label_path)

            labelPath = os.path.join(root_path +model_folderName+"/", label)

            for folder in os.listdir(labelPath):
                if folder =='.DS_Store':
                    continue

                destination_folder_img_path = os.path.join(destination_img_label_path,folder)
                destination_folder_mask_path = os.path.join(destination_masks_label_path,folder)
                destination_folder_depth_path = os.path.join(destination_depth_label_path,folder)
                destination_folder_model_path = os.path.join(destination_model_label_path,folder)

                if not os.path.exists(destination_folder_img_path):
                    os.mkdir(destination_folder_img_path)
                if not os.path.exists(destination_folder_depth_path):
                    os.mkdir(destination_folder_depth_path)
                if not os.path.exists(destination_folder_mask_path):
                    os.mkdir(destination_folder_mask_path)
                if not os.path.exists(destination_folder_model_path):
                    os.mkdir(destination_folder_model_path)

                model_folder_path = os.path.join(labelPath, folder)
                shutil.copyfile(os.path.join(model_folder_path,"model.obj"), os.path.join(destination_folder_model_path, "model.obj"))
                shutil.copyfile(os.path.join(model_folder_path,"voxel.mat"), os.path.join(destination_folder_model_path, "voxel.mat"))

                imgFolderPath  = os.path.join(labelPath, folder+'/img/')

                for filename in os.listdir(imgFolderPath):
                    for i in range(1,16):
                        if filename.__contains__(str(i)):
                            if filename.__contains__("img"):
                                shutil.copyfile(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_img_path, str(i)+".png"))

                            if filename.__contains__("depth"):
                                shutil.copyfile(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_depth_path, str(i)+".png"))

                            if filename.__contains__("layer"):
                                shutil.copyfile(os.path.join(imgFolderPath,filename), os.path.join(destination_folder_mask_path, str(i)+".png"))


    def generate_voxels(self,root_path,model_foldername='models'):
        for label in self.get_classes(root_path,model_foldername):
            labelPath = os.path.join(root_path + model_foldername+"/", label)
            for folder in os.listdir(labelPath):
                if folder =='.DS_Store':
                    continue

                model_folder_path = os.path.join(labelPath, folder)
                voxel_path = os.path.join(model_folder_path,"voxel.mat")

                interpolate_and_save(voxel_path,model_folder_path,size=32, mode='nearest')
                interpolate_and_save(voxel_path,model_folder_path,size=64,mode='nearest')
                interpolate_and_save(voxel_path,model_folder_path,size=128,mode='nearest')

if __name__ == '__main__':
    root_path = '/Users/apple/OVGU/Thesis/Dataset/pix3d/'
    destination_path = '/Users/apple/OVGU/Thesis/SynthDataset2/'

    # root = "/Users/apple/OVGU/Thesis/SynthDataset1/"
    # root='/nfs1/kprabhu/SynthDataset1/'
    root = '/Users/apple/OVGU/Thesis/Dataset/pix3d/'
    # root='/nfs1/kprabhu/pix3d/'

    #Copy file
    # Utility().copy_files_to_destination(root_path=root_path,destination_path=destination_path)
    Utility().generate_voxels(root_path=root, model_foldername='model')