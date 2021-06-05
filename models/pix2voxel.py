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
        # print("rendering_images")
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.vgg(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 26, 26])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 512, 24, 24])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 8, 8])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print("image_features")
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 8, 8])
        return image_features


class Decoder(BaseModel):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.config = config
        self.least_voxel = int(self.config.voxel_size/16)

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(2048, 512, kernel_size=4, stride=2, padding=1),
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
            torch.nn.ConvTranspose3d(32, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )
        self.weight_init()

    def forward(self, image_features):
        # print("image_features")
        # print(image_features.shape)
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.shape)
        image_features = torch.split(image_features, 1, dim=0)
        gen_volumes = []
        raw_features = []

        for features in image_features:
            # print("features")
            # print(features.shape)
            gen_volume = features.view(-1, 2048, self.least_voxel, self.least_voxel, self.least_voxel)
            # print(gen_volume.shape)
            # print(gen_volume.size())   # torch.Size([batch_size, 2048, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)
            raw_feature = gen_volume
            # print(gen_volume.size())   # torch.Size([batch_size, 8, 32, 32, 32])
            gen_volume = self.layer5(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 1, 32, 32, 32])
            raw_feature = torch.cat((raw_feature, gen_volume), dim=1)
            # print(raw_feature.size())  # torch.Size([batch_size, 9, 32, 32, 32])

            gen_volumes.append(torch.squeeze(gen_volume, dim=1))
            raw_features.append(raw_feature)

        gen_volumes = torch.stack(gen_volumes).permute(1, 0, 2, 3, 4).contiguous()
        raw_features = torch.stack(raw_features).permute(1, 0, 2, 3, 4, 5).contiguous()
        # print(gen_volumes.size())      # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(raw_features.size())     # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes

class Refiner(BaseModel):
    def __init__(self, config):
        super(Refiner, self).__init__()
        self.config = config
        self.voxel_size = config.voxel_size
        self.least_size = int(self.voxel_size/8)  # self.voxel_size divideby 2 divideby 2 divideby 2
        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32, 64, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Linear(8192, 2048),
            torch.nn.ReLU()
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Linear(2048, 8192),
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

        self.weight_init()

    def forward(self, coarse_volumes):
        # print("coarse_volumes")
        # print(coarse_volumes.shape)
        volumes_32_l = coarse_volumes.view((-1, 1, self.voxel_size, self.voxel_size, self.voxel_size))
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])

        flatten_features = self.layer4(volumes_4_l.view(-1, 128 * self.least_size * self.least_size * self.least_size))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, self.least_size, self.least_size, self.least_size)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        return volumes_32_r.view((-1, self.voxel_size, self.voxel_size, self.voxel_size))

class pix2vox(BaseModel):
    def __init__(self, config):
        super(pix2vox, self).__init__()
        self.config = config
        self.encoder = Encoder(config=config,in_channel=config.in_channels)
        self.decoder = Decoder(config=config)

        if config.pix2vox_refiner:
            self.refiner = Refiner(config=config)

    def forward(self, x):
        image_features = self.encoder(x)
        raw_features, generated_volumes = self.decoder(image_features)

        # if self.config.NETWORK.USE_MERGER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_MERGER:
        #     generated_volumes = merger(raw_features, generated_volumes)
        # else:
        #     generated_volumes = torch.mean(generated_volumes, dim=1)

        # if cfg.NETWORK.USE_REFINER and epoch_idx >= cfg.TRAIN.EPOCH_START_USE_REFINER:
        if self.config.pix2vox_refiner:
            generated_volumes = self.refiner(generated_volumes)
        #     refiner_loss = bce_loss(generated_volumes, ground_truth_volumes) * 10
        # else:
        #     refiner_loss = encoder_loss
        # print("generated_volumes")
        # print(generated_volumes.shape)
        return generated_volumes
