"""

    Created on 03/05/21 6:35 PM 
    @author: Kartik Prabhu

"""

import torch
import torch.nn as nn
import torch.utils.data

class Dice(nn.Module):
    """
    Class used to get dice_loss and dice_score
    """

    def __init__(self, smooth=1):
        super(Dice, self).__init__()
        self.smooth = smooth
        self.dicescore = 0

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_score
        self.dicescore = dice_score
        return dice_loss

    def get_dice_score(self):
        return self.dicescore


class IOU(nn.Module):
    def __init__(self, smooth=1):
        super(IOU, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred_f = torch.flatten(y_pred)
        y_true_f = torch.flatten(y_true)
        intersection = torch.sum(y_true_f * y_pred_f)
        union = torch.sum(y_true_f + y_pred_f) - intersection
        score = (intersection + self.smooth) / (union + self.smooth)
        return score


class FocalTverskyLoss(nn.Module):
    def __init__(self, smooth=1, gamma=0.75, alpha=0.7):
        super(FocalTverskyLoss, self).__init__()
        self.smooth = smooth
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        y_true_pos = torch.flatten(y_true)
        y_pred_pos = torch.flatten(y_pred)
        true_pos = torch.sum(y_true_pos * y_pred_pos)
        false_neg = torch.sum(y_true_pos * (1 - y_pred_pos))
        false_pos = torch.sum((1 - y_true_pos) * y_pred_pos)
        pt_1 = (true_pos + self.smooth) / (
                true_pos + self.alpha * false_neg + (1 - self.alpha) * false_pos + self.smooth)
        return pow((1 - pt_1), self.gamma)

class BCE(nn.Module):
    """
    Class used to get bce_loss
    """

    def __init__(self, smooth=1):
        super(BCE, self).__init__()

    def forward(self, y_pred, y_true):
        bce_loss = torch.nn.BCELoss()
        return bce_loss(y_pred,y_true)


class GANLoss(nn.Module):
    def get_disc_loss(self,real_X, fake_X, disc_X, adv_criterion):
        '''
        Return the loss of the discriminator given inputs.
        Parameters:
        real_X: the real images from pile X
        fake_X: the generated images of class X
        disc_X: the discriminator for class X; takes images and returns real/fake class X
            prediction matrices
        adv_criterion: the adversarial loss function; takes the discriminator
            predictions and the target labels and returns a adversarial
            loss (which you aim to minimize)
        '''
        disc_fake_X_hat = disc_X(fake_X.detach())
        disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
        disc_real_X_hat = disc_X(real_X)
        disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.ones_like(disc_real_X_hat))
        disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
        return disc_loss

    def get_gen_adversarial_loss(self,real_X, disc_Y, gen_XY, adv_criterion):
        '''
        Return the adversarial loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            disc_Y: the discriminator for class Y; takes images and returns real/fake class Y
            prediction matrices
            gen_XY: the generator for class X to Y; takes images and returns the images
            transformed to class Y
            adv_criterion: the adversarial loss function; takes the discriminator
                  predictions and the target labels and returns a adversarial
                  loss (which you aim to minimize)
        '''
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.ones_like(disc_fake_Y_hat))
        return adversarial_loss, fake_Y


    def get_identity_loss(self,real_X, gen_YX, identity_criterion):
        '''
        Return the identity loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
        real_X: the real images from pile X
        gen_YX: the generator for class Y to X; takes images and returns the images
            transformed to class X
        identity_criterion: the identity loss function; takes the real images from X and
                        those images put through a Y->X generator and returns the identity
                        loss (which you aim to minimize)
        '''
        identity_X = gen_YX(real_X)
        identity_loss = identity_criterion(real_X, identity_X)

        return identity_loss, identity_X

    def get_cycle_consistency_loss(self,real_X, fake_Y, gen_YX, cycle_criterion):
        '''
        Return the cycle consistency loss of the generator given inputs
        (and the generated images for testing purposes).
        Parameters:
            real_X: the real images from pile X
            fake_Y: the generated images of class Y
            gen_YX: the generator for class Y to X; takes images and returns the images
                transformed to class X
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                            those images put through a X->Y generator and then Y->X generator
                            and returns the cycle consistency loss (which you aim to minimize)
        '''
        cycle_X = gen_YX(fake_Y)
        cycle_loss = cycle_criterion(real_X,cycle_X)
        return cycle_loss, cycle_X


    def get_gen_loss(self,real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
        '''
        Return the loss of the generator given inputs.
        Parameters:
            real_A: the real images from pile A
            real_B: the real images from pile B
            gen_AB: the generator for class A to B; takes images and returns the images
                transformed to class B
            gen_BA: the generator for class B to A; takes images and returns the images
                transformed to class A
            disc_A: the discriminator for class A; takes images and returns real/fake class A
                prediction matrices
            disc_B: the discriminator for class B; takes images and returns real/fake class B
                prediction matrices
            adv_criterion: the adversarial loss function; takes the discriminator
                predictions and the true labels and returns a adversarial
                loss (which you aim to minimize)
            identity_criterion: the reconstruction loss function used for identity loss
                and cycle consistency loss; takes two sets of images and returns
                their pixel differences (which you aim to minimize)
            cycle_criterion: the cycle consistency loss function; takes the real images from X and
                those images put through a X->Y generator and then Y->X generator
                and returns the cycle consistency loss (which you aim to minimize).
                Note that in practice, cycle_criterion == identity_criterion == L1 loss
            lambda_identity: the weight of the identity loss
            lambda_cycle: the weight of the cycle-consistency loss
        '''
    # Hint 1: Make sure you include both directions - you can think of the generators as collaborating
    # Hint 2: Don't forget to use the lambdas for the identity loss and cycle loss!
    #### START CODE HERE ####

    # Adversarial Loss -- get_gen_adversarial_loss(real_X, disc_Y, gen_XY, adv_criterion)
        adversarial_loss_AB,fake_B = self.get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
        adversarial_loss_BA,fake_A = self.get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)

    # Identity Loss -- get_identity_loss(real_X, gen_YX, identity_criterion)
        identity_loss_AB,_ = self.get_identity_loss(real_A, gen_BA, identity_criterion)
        identity_loss_BA,_ = self.get_identity_loss(real_B, gen_AB, identity_criterion)

    # Cycle-consistency Loss -- get_cycle_consistency_loss(real_X, fake_Y, gen_YX, cycle_criterion)
        cycle_loss_AB,_ = self.get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
        cycle_loss_BA,_ = self.get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)

    # Total loss
        gen_loss = adversarial_loss_AB + adversarial_loss_BA + lambda_identity * (identity_loss_AB + identity_loss_BA) + lambda_cycle * (cycle_loss_AB + cycle_loss_BA)

    #### END CODE HERE ####
        return gen_loss, fake_A, fake_B