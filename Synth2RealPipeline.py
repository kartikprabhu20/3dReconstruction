"""

    Created on 05/09/21 4:56 PM 
    @author: Kartik Prabhu

"""
import os
from math import sqrt

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from torchvision.utils import save_image

from Logging import Logger
from Losses import GANLoss
from basepipeline import BasePipeline
from models.CycleGan import Generator,Discriminator
from torchvision.transforms import transforms

class BasePipeline2(BasePipeline):
    def __init__(self, config,datasetManager):
        self.datasetManager = datasetManager
        self.config = config
        self.setup_logger()

        transform = transforms.Compose([
            #     transforms.ToPILImage(),
            transforms.Resize(self.config.size[0]),
            transforms.RandomCrop(self.config.size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        self.dataset = datasetManager.get_dataset(self.config.dataset_type,logger=self.logger)

        traindataset = self.dataset .get_trainset(transforms=transform)
        self.train_dataloader = torch.utils.data.DataLoader(traindataset, batch_size=self.config.batch_size, shuffle=True,
                                                                           num_workers=self.config.num_workers, pin_memory=True)

    def setup_logger(self):
        self.logger = Logger(self.config.main_name, self.config.output_path + self.config.main_name).get_logger()
        self.logger.info("configuration: " + str(self.config))

        self.writer_training = SummaryWriter(self.config.tensorboard_train)
        self.writer_validating = SummaryWriter(self.config.tensorboard_validation)

    def tensor_images(self,image_tensor, num_images=25, nrow=5, size=(1, 28, 28)):
        '''
        Function for visualizing images: Given a tensor of images, number of images, and
        size per image, plots and prints the images in an uniform grid.
        '''
        image_tensor = (image_tensor + 1) / 2
        image_shifted = image_tensor
        image_unflat = image_shifted.detach().cpu().view(-1, *size)
        image_grid = make_grid(image_unflat[:num_images], nrow=nrow)
        # plt.imshow(image_grid.permute(1, 2, 0).squeeze())
        # plt.show()

        return image_grid


    def write_summary(self,writer, index, real_image,fake_image, mean_generator_loss,mean_discriminator_loss,path, writeImage = False):
        """
        Method to write summary to the tensorboard.
        """
        print('Writing Summary...')
        writer.add_scalar('Mean Generator Loss', mean_generator_loss, index)
        writer.add_scalar('Mean Discriminator Loss', mean_discriminator_loss, index)

        writer.add_image('Real', real_image, index)
        writer.add_image('Fake', fake_image, index)

        if writeImage:
            save_image(real_image, path+"Real_"+str(index)+".png")
            save_image(fake_image, path+"Fake_"+str(index)+".png")

class Synth2RealPipeline(BasePipeline2):
    def __init__(self, config,datasetManager):
        super(Synth2RealPipeline, self).__init__(config, datasetManager)

        self.config = config
        self.dim_A, self.dim_B = 3,3 #input,output dimension, hardcoded to 3 channels(rgb)
        self.gen_AB = Generator(self.dim_A, self.dim_B).to(config.device)
        self.gen_BA = Generator(self.dim_B, self.dim_A).to(config.device)
        self.gen_opt = torch.optim.Adam(list(self.gen_AB.parameters()) + list(self.gen_BA.parameters()), lr=config.learning_rate, betas=(0.5, 0.999))
        self.disc_A = Discriminator(self.dim_A).to(config.device)
        self.disc_A_opt = torch.optim.Adam(self.disc_A.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        self.disc_B = Discriminator(self.dim_B).to(config.device)
        self.disc_B_opt = torch.optim.Adam(self.disc_B.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))
        self.ganloss = GANLoss()
        self.adv_criterion = nn.MSELoss()
        self.recon_criterion = nn.L1Loss()

        self.display_step = self.config.save_mesh

    def train(self):

        mean_generator_loss = 0
        mean_discriminator_loss = 0
        cur_step = 0

        real_A, real_B, fake_A ,fake_B = None,None,None,None
        for epoch in range(self.config.num_epochs):
            # Dataloader returns the batches
            # for image, _ in tqdm(dataloader):
            for real_A, real_B in self.train_dataloader:
                # image_width = image.shape[3]
                real_A = nn.functional.interpolate(real_A, size=self.config.size[0])
                real_B = nn.functional.interpolate(real_B, size=self.config.size[0])
                cur_batch_size = len(real_A)
                real_A = real_A.to(self.config.device)
                real_B = real_B.to(self.config.device)

                ### Update discriminator A ###
                self.disc_A_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_A = self.gen_BA(real_B)
                disc_A_loss = self.ganloss.get_disc_loss(real_A, fake_A, self.disc_A, self.adv_criterion)
                disc_A_loss.backward(retain_graph=True) # Update gradients
                self.disc_A_opt.step() # Update optimizer

                ### Update discriminator B ###
                self.disc_B_opt.zero_grad() # Zero out the gradient before backpropagation
                with torch.no_grad():
                    fake_B = self.gen_AB(real_A)
                disc_B_loss = self.ganloss.get_disc_loss(real_B, fake_B, self.disc_B, self.adv_criterion)
                disc_B_loss.backward(retain_graph=True) # Update gradients
                self.disc_B_opt.step() # Update optimizer

                ### Update generator ###
                self.gen_opt.zero_grad()
                gen_loss, fake_A, fake_B = self.ganloss.get_gen_loss(
                    real_A, real_B, self.gen_AB, self.gen_BA, self.disc_A, self.disc_B, self.adv_criterion, self.recon_criterion, self.recon_criterion
                )
                gen_loss.backward() # Update gradients
                self.gen_opt.step() # Update optimizer

                # Keep track of the average discriminator loss
                mean_discriminator_loss += disc_A_loss.item() / self.display_step
                # Keep track of the average generator loss
                mean_generator_loss += gen_loss.item() / self.display_step

                cur_step += 1

                if cur_step % self.display_step == 0:
                    print(f"Epoch {epoch}: Step {cur_step}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                    self.logger.info("Epoch"+str(epoch)+": Step "+str(cur_step)+": Generator (U-Net) loss: "+str(mean_generator_loss)+", Discriminator loss:"+str(mean_discriminator_loss)+"}")

            ### Visualization code ###
            if epoch % self.display_step == 0:
                print(f"Epoch {epoch}: Generator (U-Net) loss: {mean_generator_loss}, Discriminator loss: {mean_discriminator_loss}")
                self.logger.info("Epoch"+str(epoch)+": Generator (U-Net) loss: "+str(mean_generator_loss)+", Discriminator loss:"+str(mean_discriminator_loss)+"}")

                grid_real = self.tensor_images(torch.cat([real_A, real_B]), size=(self.dim_A, self.config.size[0], self.config.size[0]))
                grid_fake = self.tensor_images(torch.cat([fake_B, fake_A]), size=(self.dim_B, self.config.size[0], self.config.size[0]))
                self.write_summary(self.writer_training,epoch,grid_real,grid_fake,mean_generator_loss,mean_discriminator_loss,self.config.output_path,writeImage=True)
                mean_generator_loss = 0
                mean_discriminator_loss = 0
                # You can change save_model to True if you'd like to save the model
                if epoch%50==0:

                    if not os.path.exists(self.checkpoint_path):
                        os.mkdir(self.checkpoint_path)
                        self.checkpoint_path = self.checkpoint_path + epoch+'/'
                        if not os.path.exists(self.checkpoint_path):
                            os.mkdir(self.checkpoint_path)

                    torch.save({
                        'gen_AB': self.gen_AB.state_dict(),
                        'gen_BA': self.gen_BA.state_dict(),
                        'gen_opt': self.gen_opt.state_dict(),
                        'disc_A': self.disc_A.state_dict(),
                        'disc_A_opt': self.disc_A_opt.state_dict(),
                        'disc_B': self.disc_B.state_dict(),
                        'disc_B_opt': self.disc_B_opt.state_dict()
                    }, self.checkpoint_path + 'checkpoint_' + epoch + '.pth')





