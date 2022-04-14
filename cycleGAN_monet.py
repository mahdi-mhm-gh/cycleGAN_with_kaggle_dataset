import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import torchvision.transforms as transforms
from tqdm.notebook import tqdm

transforms_ = transforms.Compose([
    transforms.Resize(int(256*1.12), Image.BICUBIC),
    transforms.RandomCrop((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class image_dataset(Dataset):
    def __init__(self,transform = None, data_type = None):
        self.transform = transform
        self.data_type = data_type
        root_dir = "../input/gan-getting-started"
        if data_type == "train":
            self.style_image = sorted(glob.glob(os.path.join(root_dir,"monet_jpg"+"/*.*")))[:250]
            self.content_image = sorted(glob.glob(os.path.join(root_dir,"photo_jpg"+"/*.*")))[:250]
        elif data_type == "test":
            self.style_image = sorted(glob.glob(os.path.join(root_dir,"monet_jpg"+"/*.*")))[250:]
            self.content_image = sorted(glob.glob(os.path.join(root_dir,"photo_jpg"+"/*.*")))[250:301]
    def __len__(self):
        return max(len(self.style_image), len(self.content_image))
    def __getitem__(self, index):
        style_image = Image.open(self.style_image[index % len(self.style_image)])
        content_image = Image.open(self.content_image[index % len(self.style_image)])
        if self.transform:
            image_c = self.transform(content_image)
            image_s = self.transform(style_image)
        return image_c, image_s
      
train_dataset = image_dataset(transform = transforms_, data_type = "train")
test_dataset = image_dataset(transform = transforms_, data_type ="test")
train_loader = DataLoader(train_dataset, batch_size = 8,shuffle = True)
test_loader = DataLoader(test_dataset, batch_size = 5, shuffle = True)


class CovBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode="reflect", **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
           CovBlock(channels, channels, kernel_size=3, padding=1),
           CovBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, img_channels, num_features = 64, num_residuals=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode="reflect"),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList(
            [
                CovBlock(num_features, num_features*2, kernel_size=3, stride=2, padding=1),
                CovBlock(num_features*2, num_features*4, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList(
            [
                CovBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                CovBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, padding_mode="reflect")

    def forward(self, x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
      
 class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode="reflect",
            ),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))
      
  def sample_images():
    """show a generated sample from the test set"""
    for image_c, image_s in test_loader:
        gen_monet.eval()
        gen_photo.eval()
        image_c = image_c.cuda()
        fake_monet = gen_monet(image_c).detach()
        image_s = image_s.cuda() 
        fake_photo = gen_photo(image_s).detach()
        # Arange images along x-axis
        photo = make_grid(image_c, nrow=5, normalize=True)
        fake_monet = make_grid(fake_monet, nrow=5, normalize=True)
        monet = make_grid(image_s, nrow=5, normalize=True)
        fake_photo = make_grid(fake_photo, nrow=5, normalize=True)
        # Arange images along y-axis    
        image_grid = torch.cat((photo, fake_monet, monet, fake_photo), 1)
        plt.imshow(image_grid.cpu().permute(1,2,0))
        plt.title('photo A vs fake_monet| monet B vs fake_photo')
        plt.axis('off')
        plt.show()
        break
        
gen_monet = Generator(3, num_residuals = 9)
gen_photo = Generator(3, num_residuals = 9)
disc_monet = Discriminator(in_channels = 3)
disc_photo = Discriminator(in_channels = 3)

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02) 
        if hasattr(m, 'bias') and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0) 
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02) 
            torch.nn.init.constant_(m.bias.data, 0.0)
            
gen_monet.apply(weights_init_normal)
gen_photo.apply(weights_init_normal)
disc_monet.apply(weights_init_normal)
disc_photo.apply(weights_init_normal)

l1 = nn.L1Loss()
mse = nn.MSELoss()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 10 

gen_monet.cuda()
gen_photo.cuda()
disc_monet.cuda()
disc_photo.cuda()
l1.cuda()
mse.cuda()

opt_gen = torch.optim.Adam(list(gen_monet.parameters()) + list(gen_photo.parameters()),lr=0.0002,betas=(0.5, 0.999))
opt_disc = torch.optim.Adam(list(disc_monet.parameters()) + list(disc_photo.parameters()),lr=0.0002,betas=(0.5, 0.999))

g_scaler = torch.cuda.amp.GradScaler()
d_scaler = torch.cuda.amp.GradScaler()
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
for epoch in range(epochs):
    for i, (photo, monet) in enumerate(tqdm(train_loader)):
        photo = photo.cuda()
        monet = monet.cuda()
        gen_monet.train()
        gen_photo.train()
        with torch.cuda.amp.autocast():
            fake_monet = gen_monet(photo)
            fake_photo = gen_photo(monet)
            cycle_monet = gen_monet(fake_photo)
            cycle_photo = gen_photo(fake_monet)
            cycle_monet_loss = l1(cycle_monet, monet)
            cycle_photo_loss = l1(cycle_photo, photo)
            identity1= gen_monet(monet)
            identitiy2 = gen_photo(photo)
            identity1_loss = l1(identity1, monet)
            identity2_loss = l1(identitiy2, photo)
            gen_monet_loss = mse(fake_monet, torch.ones_like(fake_monet))
            gen_photo_loss = mse(fake_photo, torch.ones_like(fake_photo))
            G_loss = (cycle_monet_loss * 10.0 + cycle_photo_loss * 10.0 
                      + identity1_loss * 5.0 + identity2_loss * 5.0  
                      + gen_monet_loss + gen_photo_loss)
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()
        with torch.cuda.amp.autocast():
            disc_fmonet = disc_monet(fake_monet.detach())
            disc_fphoto = disc_photo(fake_photo.detach())
            disc_rmonet = disc_monet(monet)
            disc_rphoto = disc_photo(photo)
            discfm_loss = mse(disc_fmonet, torch.zeros_like(disc_fmonet))
            discrm_loss = mse(disc_rmonet, torch.ones_like(disc_rmonet))
            disc_monet_loss = (discfm_loss + discrm_loss)/2
            discrp_loss = mse(disc_rphoto, torch.ones_like(disc_rphoto))
            discfp_loss = mse(disc_fphoto, torch.zeros_like(disc_fphoto))
            disc_photo_loss = (discrp_loss + discfp_loss)/2
            D_loss = disc_photo_loss + disc_monet_loss
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()
        
    sample_images()
