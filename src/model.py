import torch
import wandb
import torch.nn as nn
from torch.nn.functional import elu, instance_norm

import pytorch_lightning as pl


class SkipConnection(nn.Module):
    def forward(self, out, old_out):
        return torch.cat([out, old_out], dim=1)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Clip(nn.Module):
    def forward(self, input):
        return torch.clamp(input, min=-1, max=1)


class GatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,):
        super(GatedConv, self).__init__()
        self.conv2d = nn.Conv2d(in_channels,
                                out_channels,
                                kernel_size,
                                stride,
                                padding,
                                dilation,
                                groups,
                                bias)
        self.mask_conv2d = nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size,
                                     stride,
                                     padding,
                                     dilation,
                                     groups,
                                     bias)
        self.sigmoid = nn.Sigmoid()

    def gated(self, mask):
        return self.sigmoid(mask)

    def forward(self, input):
        x = self.conv2d(input)
        mask = self.mask_conv2d(input)
        x = elu(x) * self.gated(mask)
        x = instance_norm(x)
        return x


class Generator(pl.LightningModule):
    def __init__(self):
        super(Generator, self).__init__()
        self.skip = SkipConnection()
        self.upsampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        self.cache_GC_1 = GatedConv(4, 32, 5, 1, 2)
        self.cache_GC_2 = GatedConv(32, 64, 3, 2, 1)
        self.cache_GC_3 = GatedConv(64, 64, 3, 1, 1)
        self.cache_GC_4 = GatedConv(64, 128, 3, 2, 1)
        self.cache_GC_5 = GatedConv(128, 128, 3, 1, 1)

        self.mid_pile = nn.Sequential(
            GatedConv(128, 128, 3, 1, 1),
            GatedConv(128, 128, 3, 1, 2, dilation=2),
            GatedConv(128, 128, 3, 1, 4, dilation=4),
            GatedConv(128, 128, 3, 1, 8, dilation=8), 
            GatedConv(128, 128, 3, 1, 16, dilation=16),
            GatedConv(128, 128, 3, 1, 1)
        )

        self.GC_1 = GatedConv(256, 128, 3, 1, 1)
        self.GC_2 = GatedConv(256, 64, 3, 1, 1)
        self.GC_3 = GatedConv(128, 64, 3, 1, 1)
        self.GC_4 = GatedConv(128, 32, 3, 1, 1)
        self.GC_5 = GatedConv(64, 16, 3, 1, 1)

        self.final_conv = nn.Conv2d(16, 3, kernel_size=3, stride=1, padding=1)
        self.clip = Clip()

    def forward(self, x):
        out1 = self.cache_GC_1(x)
        out2 = self.cache_GC_2(out1)
        out3 = self.cache_GC_3(out2)
        out4 = self.cache_GC_4(out3)
        out5 = self.cache_GC_5(out4)

        out = self.mid_pile(out5)

        out = self.skip(out, out5)
        out = self.GC_1(out)

        out = self.skip(out, out4)
        out = self.upsampling(out)
        out = self.GC_2(out)

        out = self.skip(out, out3)
        out = self.GC_3(out)

        out = self.skip(out, out2)
        out = self.upsampling(out)
        out = self.GC_4(out)

        out = self.skip(out, out1)
        out = self.GC_5(out)

        out = self.final_conv(out)
        out = self.clip(out)
        return out


class Discriminator(pl.LightningModule):
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = []
        in_channels, out_channels = 4, 64
        for _ in range(6):
            layers.extend(
                (
                    nn.utils.spectral_norm(
                        nn.Conv2d(
                            in_channels,
                            out_channels,
                            kernel_size=5,
                            stride=2,
                            padding=2,
                        )
                    ),
                    nn.LeakyReLU(),
                )
            )
            in_channels = out_channels
            out_channels = 2 * out_channels if out_channels < 256 else out_channels
        layers.extend(
            (
                nn.utils.spectral_norm(
                    nn.Conv2d(256, 256, kernel_size=4, stride=1, padding=0)
                ),
                nn.LeakyReLU(),
            )
        )
        self.pile = nn.Sequential(*layers)

        self.flatten = Flatten()
        self.cond_linear = nn.utils.spectral_norm(nn.Linear(1000, 256, bias=False))
        self.final_linear = nn.utils.spectral_norm(nn.Linear(256, 1, bias=False))

    def forward(self, x, y, z):
        out = torch.cat([x, y], dim=1)
        out = self.pile(out)
        out = self.flatten(out)
        out_t = self.final_linear(out)

        z = self.cond_linear(z)
        out = (out * z).sum(1, keepdim=True)
        out = torch.add(out, out_t)
        return out

class Boundless_GAN(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args

        self.generator = Generator()
        self.discriminator = Discriminator()
        self.automatic_optimization = False

    def forward(self, z):
        return self.generator(z)

    def generator_step(self, input_tensor, real_image, masked_image, mask, cond):

        gen_image = self.generator(input_tensor)
        loss_rec = torch.nn.L1Loss()(gen_image, real_image)

        substituted_gen_image = gen_image * mask + masked_image
        loss_adv = -self.discriminator(substituted_gen_image, mask, cond).mean()

        loss_G = self.args.lambda_adv * loss_adv + loss_rec

        return {
            'loss_G': loss_G,
            "loss_adv": loss_adv,
            'loss_rec':  loss_rec
        }

    def discriminator_step(self, real_image, fake_image, mask, cond):
        pred_real = self.discriminator(real_image, mask, cond)
        pred_fake = self.discriminator(fake_image.detach(), mask, cond)
        loss_D = nn.ReLU()(1.0 - pred_real).mean() + nn.ReLU()(1.0 + pred_fake).mean()

        return {
            'loss_D': loss_D
        }

    def training_step(self, batch, batch_idx):
        optimizer_g, optimizer_d = self.optimizers()

        input_tensor = torch.Tensor(batch["input_tensor"])
        cond = torch.Tensor(batch["inception_embeds"])
        
        real_image = input_tensor[:, :3, :, :]
        mask = input_tensor[:, 3:, :, :]
        masked_image = mask * real_image

        
        self.toggle_optimizer(optimizer_g)

        G_output = self.generator_step(input_tensor, real_image, masked_image, mask, cond)
        loss_g =  G_output['loss_G']
        self.log('Generator loss', loss_g)
        self.manual_backward(loss_g)
        optimizer_g.step()
        optimizer_g.zero_grad()
        self.untoggle_optimizer(optimizer_g)

        self.toggle_optimizer(optimizer_d)

        gen_image = self.generator(input_tensor)
        fake_image = gen_image * mask + masked_image

        D_output = self.discriminator_step(real_image, fake_image, mask, cond)
        loss_d = D_output['loss_D']
        self.log('Discriminator loss', loss_d)
        self.manual_backward(loss_d)
        optimizer_d.step()
        optimizer_d.zero_grad()
        self.untoggle_optimizer(optimizer_d)

        if batch_idx % self.args.log_every == 0:
            image = wandb.Image(fake_image.cpu().detach().numpy().transpose(0, 2, 3, 1), caption="Fake image")
            wandb.log({"example":  image})
            wandb.log({'Current generator loss': loss_g, 'Current discriminator loss': loss_d})

    def configure_optimizers(self):
        optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.args.lr_g, betas=(self.args.b1, self.args.b2))
        optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.args.lr_d, betas=(self.args.b1, self.args.b2))
        return [optimizer_G, optimizer_D], []
