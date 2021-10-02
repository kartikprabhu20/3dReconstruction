"""

    Created on 13/08/21 11:38 PM 
    @author: Kartik Prabhu

"""

import torch
import torchvision.models
import torch.nn as nn

import utils

class BaseModel(nn.Module):
    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                if isinstance(m, (nn.Linear, nn.Conv2d)):
                    torch.nn.init.xavier_uniform_(m.weight)
                elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.weight.data.fill_(1)

class Encoder(torch.nn.Module):
    def __init__(self, cfg, in_channel = 3):
        super(Encoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        resnet = torchvision.models.resnet50(pretrained=self.cfg.pix2vox_pretrained_encoder)
        self.resnet = torch.nn.Sequential(*[
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3,
            resnet.layer4
        ])[:6]
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, rendering_images):
        # print(rendering_images.size())  # torch.Size([batch_size, n_views, img_c, img_h, img_w])
        rendering_images = rendering_images.permute(1, 0, 2, 3, 4).contiguous()
        rendering_images = torch.split(rendering_images, 1, dim=0)
        image_features = []

        for img in rendering_images:
            features = self.resnet(img.squeeze(dim=0))
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer1(features)
            # print(features.size())    # torch.Size([batch_size, 512, 28, 28])
            features = self.layer2(features)
            # print(features.size())    # torch.Size([batch_size, 256, 14, 14])
            features = self.layer3(features)
            # print(features.size())    # torch.Size([batch_size, 256, 7, 7])
            image_features.append(features)

        image_features = torch.stack(image_features).permute(1, 0, 2, 3, 4).contiguous()
        # print(image_features.size())  # torch.Size([batch_size, n_views, 256, 7, 7])
        return image_features

class Decoder(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1568, 512, kernel_size=4, stride=2, padding=1),
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

        self.layer4_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.layer4_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 8, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(8),
            torch.nn.ReLU()
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(8, 1, kernel_size=1),
            torch.nn.Sigmoid()
        )


    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        raw_features = []
        gen_volumes = []

        for features in image_features:
            gen_volume = features.view(-1, 1568, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 1568, 2, 2, 2])
            gen_volume = self.layer1(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 512, 4, 4, 4])
            gen_volume = self.layer2(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 128, 8, 8, 8])
            gen_volume = self.layer3(gen_volume)
            # print(gen_volume.size())   # torch.Size([batch_size, 32, 16, 16, 16])
            gen_volume = self.layer4(gen_volume)

            if self.cfg.voxel_size == 64 or self.cfg.voxel_size == 128:
                gen_volume = self.layer4_1(gen_volume)
            if self.cfg.voxel_size == 128:
                gen_volume = self.layer4_2(gen_volume)

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
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes


class Decoder2(torch.nn.Module):
    def __init__(self, cfg):
        super(Decoder2, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(1568, 512, kernel_size=4, stride=2, padding=1),
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


    def forward(self, image_features):
        image_features = image_features.permute(1, 0, 2, 3, 4).contiguous()
        image_features = torch.split(image_features, 1, dim=0)
        raw_features = []
        gen_volumes = []

        for features in image_features:
            gen_volume = features.view(-1, 1568, 2, 2, 2)
            # print(gen_volume.size())   # torch.Size([batch_size, 1568, 2, 2, 2])
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
        # print(raw_features.size())      # torch.Size([batch_size, n_views, 9, 32, 32, 32])
        return raw_features, gen_volumes

class Merger(torch.nn.Module):
    def __init__(self, cfg):
        super(Merger, self).__init__()
        self.cfg = cfg

        # Layer Definition
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv3d(36, 9, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(9),
            torch.nn.LeakyReLU(0.2)
        )
        self.layer6 = torch.nn.Sequential(
            torch.nn.Conv3d(9, 1, kernel_size=3, padding=1),
            torch.nn.BatchNorm3d(1),
            torch.nn.LeakyReLU(0.2)
        )

    def forward(self, raw_features, coarse_volumes):
        n_views_rendering = coarse_volumes.size(1)
        raw_features = torch.split(raw_features, 1, dim=1)
        volume_weights = []

        for i in range(n_views_rendering):
            raw_feature = torch.squeeze(raw_features[i], dim=1)
            # print(raw_feature.size())       # torch.Size([batch_size, 9, 32, 32, 32])
            volume_weight1 = self.layer1(raw_feature)
            # print(volume_weight1.size())    # torch.Size([batch_size, 9, 32, 32, 32])
            volume_weight2 = self.layer2(volume_weight1)
            # print(volume_weight2.size())    # torch.Size([batch_size, 9, 32, 32, 32])
            volume_weight3 = self.layer3(volume_weight2)
            # print(volume_weight3.size())    # torch.Size([batch_size, 9, 32, 32, 32])
            volume_weight4 = self.layer4(volume_weight3)
            # print(volume_weight4.size())    # torch.Size([batch_size, 9, 32, 32, 32])
            volume_weight = self.layer5(torch.cat([
                volume_weight1, volume_weight2, volume_weight3, volume_weight4
            ], dim=1))
            # print(volume_weight.size())     # torch.Size([batch_size, 9, 32, 32, 32])
            volume_weight = self.layer6(volume_weight)
            # print(volume_weight.size())     # torch.Size([batch_size, 1, 32, 32, 32])
            volume_weight = torch.squeeze(volume_weight, dim=1)
            # print(volume_weight.size())     # torch.Size([batch_size, 32, 32, 32])
            volume_weights.append(volume_weight)

        volume_weights = torch.stack(volume_weights).permute(1, 0, 2, 3, 4).contiguous()
        volume_weights = torch.softmax(volume_weights, dim=1)
        # print(volume_weights.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        # print(coarse_volumes.size())        # torch.Size([batch_size, n_views, 32, 32, 32])
        coarse_volumes = coarse_volumes * volume_weights
        coarse_volumes = torch.sum(coarse_volumes, dim=1)

        return torch.clamp(coarse_volumes, min=0, max=1)

class Refiner(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner, self).__init__()
        self.cfg = cfg

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

        self.layer3_1 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=4, padding=2),
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

        self.layer6_1 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm3d(128),
            torch.nn.ReLU()
        )

        self.layer6_2 = torch.nn.Sequential(
            torch.nn.ConvTranspose3d(128, 128, kernel_size=4, stride=2, padding=1),
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
        volumes_32_l = coarse_volumes.unsqueeze(dim=1)
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        return volumes_32_r.squeeze(dim=1)



class Refiner2(torch.nn.Module):
    def __init__(self, cfg):
        super(Refiner2, self).__init__()
        self.cfg = cfg

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

        self.layer3_1 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=4, padding=2),
            torch.nn.BatchNorm3d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.MaxPool3d(kernel_size=2)
        )

        self.layer3_2 = torch.nn.Sequential(
            torch.nn.Conv3d(128, 128, kernel_size=4, padding=2),
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

    def forward(self, coarse_volumes):
        volumes_32_l = coarse_volumes.unsqueeze(dim=1)
        # print(volumes_32_l.size())       # torch.Size([batch_size, 1, 32, 32, 32])
        volumes_16_l = self.layer1(volumes_32_l)
        # print(volumes_16_l.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_8_l = self.layer2(volumes_16_l)
        # print(volumes_8_l.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_4_l = self.layer3(volumes_8_l)
        # print(volumes_4_l.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        flatten_features = self.layer4(volumes_4_l.view(-1, 8192))
        # print(flatten_features.size())   # torch.Size([batch_size, 2048])
        flatten_features = self.layer5(flatten_features)
        # print(flatten_features.size())   # torch.Size([batch_size, 8192])
        volumes_4_r = volumes_4_l + flatten_features.view(-1, 128, 4, 4, 4)
        # print(volumes_4_r.size())        # torch.Size([batch_size, 128, 4, 4, 4])
        volumes_8_r = volumes_8_l + self.layer6(volumes_4_r)
        # print(volumes_8_r.size())        # torch.Size([batch_size, 64, 8, 8, 8])
        volumes_16_r = volumes_16_l + self.layer7(volumes_8_r)
        # print(volumes_16_r.size())       # torch.Size([batch_size, 32, 16, 16, 16])
        volumes_32_r = (volumes_32_l + self.layer8(volumes_16_r)) * 0.5
        # print(volumes_32_r.size())       # torch.Size([batch_size, 1, 32, 32, 32])

        return volumes_32_r.squeeze(dim=1)


class pix2voxpp(BaseModel):
    def __init__(self, config):
        super(pix2voxpp, self).__init__()
        self.config = config
        self.encoder = Encoder(cfg=config,in_channel=config.in_channels)
        self.decoder = Decoder(cfg=config)
        self.decoder2 = Decoder2(cfg=config)

        print('Parameters in Encoder: %d.' % (utils.count_parameters(self.encoder)))
        print('Parameters in Decoder: %d.' % (utils.count_parameters(self.decoder)))

        if config.pix2vox_refiner:
            self.refiner = Refiner(cfg=config)
            self.refiner2 = Refiner2(cfg=config)
            print('Parameters in Refiner: %d.' % (utils.count_parameters(self.refiner)))

        if config.pix2vox_merger:
            self.merger =  Merger(cfg=config)
            print('Parameters in Merger: %d.' % (utils.count_parameters(self.merger)))

    def forward(self, x):
        image_features = self.encoder(x)
        dec_raw_features, dec_generated_volumes = self.decoder(image_features)

        if self.config.pix2vox_merger:
            dec_generated_volumes = self.merger(dec_raw_features, dec_generated_volumes)
        else:
            dec_generated_volumes = torch.mean(dec_generated_volumes, dim=1)

        if self.config.pix2vox_refiner:
            ref_generated_volumes = self.refiner(dec_generated_volumes)
        else:
            ref_generated_volumes = None

        return dec_generated_volumes,ref_generated_volumes