"""

    Created on 20/04/21 6:28 AM
    @author: Kartik Prabhu

"""

import torch
import torchvision.models
import torch.nn as nn

class BaseModel(nn.Module):
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.weight.data.fill_(1)

class Encoder(BaseModel):
    def __init__(self,config, in_channel = 3):
        super(Encoder, self).__init__()

        # Layer Definition
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(in_channel, 3, kernel_size=1))
        vgg16_bn = torchvision.models.vgg16_bn(pretrained=config.pix2vox_pretrained_vgg)
        self.vgg = torch.nn.Sequential(*list(vgg16_bn.features.children()))[:27]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3),
            torch.nn.BatchNorm2d(512),
            torch.nn.ELU(),
            torch.nn.MaxPool2d(kernel_size=3)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ELU()
        )

        # Don't update params in VGG16
        for param in vgg16_bn.parameters():
            param.requires_grad = config.pix2vox_update_vgg

        self.weight_init()


    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        # print("-----rendering_images-----")
        # print(rendering_images.shape)
        # rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        # print(rendering_images.shape)
        # rendering_images = torch.split(rendering_images, 1, dim=0)
        # print(rendering_images.shape)
        # print("======rendering_images======")

        # image_features = []

        # for img in rendering_images:
        # print(rendering_images.size())
        img = self.conv1(rendering_images)#reshape for vgg
        # print(img.shape)

        features = self.vgg(img.squeeze(dim=0))
        # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
        features = self.layer1(features)
        # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
        features = self.layer2(features)
        # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
        features = self.layer3(features)
        # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
        # image_features.append(features)

        # image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        # return image_features
        return features

class Decoder(BaseModel):
    def __init__(self):
        super(Decoder, self).__init__()

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1600, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(512, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(16),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(16, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

        self.weight_init()

    def forward(self, image_features):
        # print("image_features")
        # print(image_features.shape)
        # image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        # image_features = torch.split(image_features, 1, dim=0)
        gen_volumes = []
        raw_features = []

        # for features in image_features:

        gen_volume = image_features.view(-1, 1600, 4, 4, 4)
        # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])# torch.Size([batch_size, 1600, 4, 4, 4])
        gen_volume = self.layer1(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
        gen_volume = self.layer2(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
        gen_volume = self.layer3(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
        gen_volume = self.layer4(gen_volume)
        # raw_feature = gen_volume
        # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
        gen_volume = self.layer5(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        # raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
        # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])

        # gen_volume = self.layer6(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        # gen_volume = self.layer7(gen_volume)
        # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
        # gen_volumes.append(torch.squeeze(gen_volume, dim=1))
        # raw_features.append(raw_feature)

        # gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        # raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())     # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        # print(gen_volume[:,0,:].shape)
        return gen_volume[:,0,:]

class Refiner(torch.nn.Module):
    def __init__(self):
        super(Refiner, self).__init__()

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 256, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(256),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.layer3_3 = torch.nn.Sequential(
            torch.nn.Conv3d(256, 512, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(512),
            torch.nn.LeakyReLU(.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(512*4*4*4, 512),
            torch.nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(512, 512*4*4*4),
            torch.nn.ReLU()
        )


        self.layer6_3 = torch.nn.Sequential(
        torch.nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1),
        torch.nn.BatchNorm3d(256),
        torch.nn.ReLU()
        )

        self.layer6_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )

        self.layer6 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(64),
            torch.nn.ReLU()
        )
        self.layer7 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(32),
            torch.nn.ReLU()
        )
        self.layer8 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(32, 1, kernel_size=4, stride=2, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, coarse_volumes):
        volumes_128_l = coarse_volumes.view((-1, 1, 128, 128, 128)) # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_64_l = self.layer1(volumes_128_l) # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_32_l = self.layer2(volumes_64_l)  # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_l = self.layer3(volumes_32_l)
        volumes_8_l = self.layer3_2(volumes_16_l)
        volumes_4_l = self.layer3_3(volumes_8_l)

        flatten_features = self.layer4(volumes_4_l.view(-1, 512*4*4*4))# torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)  # torch.Size([batch_size, 8192])

        volumes_4_r = volumes_4_l + flatten_features.view(-1, 512, 4, 4, 4)
        volumes_8_r = volumes_8_l + self.layer6_3(volumes_4_r)
        volumes_16_r = volumes_16_l + self.layer6_2(volumes_8_r) # torch.Size([batch_size, 128, 4, 4, 4])
        volumes_32_r = volumes_32_l + self.layer6(volumes_16_r) # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_64_r = volumes_64_l + self.layer7(volumes_32_r) # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_128_r = (volumes_128_l + self.layer8(volumes_64_r)) * 0.5 # torch.Size([batch_size, 1, 32, 32, 32])

        return volumes_128_r.view((-1, 128, 128,128))

class pix2vox(BaseModel):
    def __init__(self, config):
        super(pix2vox, self).__init__()

        modules = [Encoder(config=config,in_channel=config.in_channels),
                   Decoder()]

        if config.pix2vox_refiner:
            modules.append(Refiner())

        self.model = nn.Sequential(*modules)
        self.weight_init()

    def forward(self, x):

        return self.model(x)
