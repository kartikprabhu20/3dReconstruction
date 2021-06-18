"""

    Created on 10/05/21 8:36 AM 
    @author: Kartik Prabhu

"""
import io
import os

import PIL
import torch
from pytorch3d.ops import cubify
import numpy as np
from skimage.filters import threshold_otsu
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor

import Baseconfig
import ModelManager
import transform_utils
from Logging import Logger
from Losses import Dice, BCE, FocalTverskyLoss, IOU
from ModelManager import LossTypes
from utils import create_binary, save_obj
import matplotlib.pyplot as plt
import transform_utils
# from torchvision import transforms

class BasePipeline:
    def __init__(self,model,optimizer, config,datasetManager):
        self.config = config
        self.setup_logger()
        self.setup_dataloaders(datasetManager)

        self.model = model
        self.optimizer = optimizer
        self.checkpoint_path = config.checkpoint_path
        self.train_loss = self.get_loss(config.train_loss_type)
        self.train_loss_is_bce = config.train_loss_type == ModelManager.LossTypes.BCE
        self.num_epochs = config.num_epochs
        self.with_apex = config.apex

    def setup_dataloaders(self,datasetManager):
        IMG_SIZE = self.config.size[0],self.config.size[1]
        CROP_SIZE = self.config.crop_size[0], self.config.crop_size[1]

        train_transforms = transform_utils.Compose([
            transform_utils.RandomCrop(IMG_SIZE, CROP_SIZE),
            transform_utils.RandomBackground(self.config.TRAIN.RANDOM_BG_COLOR_RANGE),
            transform_utils.ColorJitter(self.config.TRAIN.BRIGHTNESS, self.config.TRAIN.CONTRAST, self.config.TRAIN.SATURATION),
            transform_utils.RandomNoise(self.config.TRAIN.NOISE_STD),
            transform_utils.Normalize(mean=self.config.DATASET.MEAN, std=self.config.DATASET.STD),
            transform_utils.RandomFlip(),
            transform_utils.RandomPermuteRGB(),
            transform_utils.ToTensor(),
        ])

        test_transforms = transform_utils.Compose([
             transform_utils.CenterCrop(IMG_SIZE, CROP_SIZE),
             transform_utils.RandomBackground(self.config.TEST.RANDOM_BG_COLOR_RANGE),
             transform_utils.Normalize(mean=self.config.DATASET.MEAN, std=self.config.DATASET.STD),
             transform_utils.ToTensor(),
         ])


        dataset = datasetManager.get_dataset(self.config.dataset_type)

        traindataset = dataset.get_trainset(transforms=train_transforms)
        self.train_loader = torch.utils.data.DataLoader(traindataset, batch_size=self.config.batch_size, shuffle=True,
                                               num_workers=self.config.num_workers, pin_memory=True)


        testdataset = dataset.get_testset(transforms=test_transforms)
        self.test_loader = torch.utils.data.DataLoader(testdataset, batch_size=self.config.batch_size, shuffle=False,
                                              num_workers=self.config.num_workers,pin_memory=True,)

    def setup_logger(self):
        self.logger = Logger(self.config.main_name, self.config.output_path + self.config.main_name).get_logger()
        self.logger.info("configuration: " + str(self.config))

        self.writer_training = SummaryWriter(self.config.tensorboard_train)
        self.writer_validating = SummaryWriter(self.config.tensorboard_validation)

    def get_loss(self,loss_type):
        switcher = {
            LossTypes.DICE: Dice(),
            LossTypes.BCE: BCE(),
            LossTypes.FOCAL_TVERSKY: FocalTverskyLoss(),
            LossTypes.IOU: IOU()
        }
        return switcher.get(loss_type, BCE())

    def gen_plot(self, voxels, savefig = False, path = None):
        """Create a pyplot plot and save to buffer."""
        ax = plt.figure().add_subplot(projection='3d')
        ax.voxels(voxels,facecolors='gray',edgecolor='k')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.axis('off')
        if savefig:
            plt.savefig(path)
        buf.seek(0)
        plt.close('all')
        return buf

    def plot_to_tensor(self, plot_buf):
        image = PIL.Image.open(plot_buf)
        return  ToTensor()(image)

    def save_intermediate_obj(self, istrain, training_batch_index, local_labels, outputs):
        process = "train" if istrain else "val"
        with torch.no_grad():

            output_path = self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_output.npy"
            #Pytorch error to cubify output from model, need to save it in np and then convert
            np.save(output_path, outputs[0].cpu().data.numpy())
            np.save(self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original.npy", local_labels[0].cpu().data.numpy())

            save_count = min(self.config.save_count,self.config.batch_size)
            for i in range(0,save_count):
                self.gen_plot(local_labels[i],savefig=True,path = self.checkpoint_path + self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original_"+str(i)+".png")
                output_np = outputs[i].detach().cpu().numpy()
                output_np = (output_np > threshold_otsu(output_np)).astype(int)
                output = torch.tensor(output_np)
                self.gen_plot(output, savefig=True, path =self.checkpoint_path + self.config.main_name + "_" + str(training_batch_index) + "_" + process + "_output_" + str(i) + ".png")

            # mesh_np = np.load(output_path)
            # thresh = threshold_otsu(mesh_np)
            # print(thresh)
            # mesh_np = (mesh_np > thresh).astype(int)
            # mesh_mat = torch.tensor(mesh_np)[None,]


            # print(mesh_mat.shape)
            # print(local_labels.shape)
            # output_mesh = cubify(mesh_mat, thresh=0.5)
            # label_mesh = cubify(local_labels[0][None,:],thresh=0.5)
            #
            # save_obj(self.checkpoint_path +self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_output.obj",verts=output_mesh.verts_list()[0], faces=output_mesh.faces_list()[0])
            # save_obj(self.checkpoint_path +self.config.main_name+"_"+ str(training_batch_index)+"_"+process+"_original.obj",verts=label_mesh.verts_list()[0], faces=label_mesh.faces_list()[0])

    def write_summary(self,writer, index, input_image, input_voxel, output_voxel, bceloss, diceLoss, diceScore, iou, writeMesh = False):
        """
        Method to write summary to the tensorboard.
        index: global_index for the visualisation
        original,reconstructer: cubified voxels [channel, Height, Width]
        Losses: all losses used as metric
        """
        print('Writing Summary...')
        writer.add_scalar('BCELoss', bceloss, index)
        writer.add_scalar('DiceLoss', diceLoss, index)
        writer.add_scalar('DiceScore', diceScore, index)
        writer.add_scalar('IOU', iou, index)

        if writeMesh:
            writer.add_image('input_image', input_image, index)
            writer.add_image('input_voxels', self.plot_to_tensor(self.gen_plot(input_voxel.detach().cpu().numpy())), index)

            output_np = output_voxel.detach().cpu().numpy()
            output_np = (output_np > threshold_otsu(output_np)).astype(int)
            output = torch.tensor(output_np)
            writer.add_image('output_voxels', self.plot_to_tensor(self.gen_plot(output)), index)

        # if writeMesh:
        #     writer.add_mesh('label', vertices=original.verts_list()[0][None,], faces=original.faces_list()[0][None,], global_step=index)
        #     writer.add_mesh('reconstructed', vertices=reconstructed.verts_list()[0][None,], faces=reconstructed.faces_list()[0][None,], global_step=index)

        # writer.add_image('diff', np.moveaxis(create_diff_mask(reconstructed,original,logger), -1, 0), index) #create_diff_mask is of the format HXWXC, but CXHXW is needed

    def save_model(self,CHECKPOINT_PATH, state, best_metric = False,filename='checkpoint'):
        """
        Method to save model
        """
        print('Saving model...')
        if not os.path.exists(CHECKPOINT_PATH):
            os.mkdir(CHECKPOINT_PATH)
            if best_metric:
                if not os.path.exists(CHECKPOINT_PATH + 'best_metric/'):
                    CHECKPOINT_PATH = CHECKPOINT_PATH + 'best_ metric/'
                    os.mkdir(CHECKPOINT_PATH)
                    torch.save(state, CHECKPOINT_PATH + filename + str(state['epoch']) + '.pth')
