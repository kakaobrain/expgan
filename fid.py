import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import torchvision.transforms as T
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance as FID
# from torchmetrics.image.fid import FID
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--root_real', type=str)
parser.add_argument('--root_fake', type=str)
parser.add_argument('--batch_size', type=int)

args = parser.parse_args()

BATCH_SIZE = args.batch_size
ROOT_REAL = args.root_real
ROOT_FAKE = args.root_fake


class MetricDataset(Dataset) : 
    def __init__(self, real_dir, fake_dir) : 

        self.real_files = sorted(glob(os.path.join(real_dir, '*.png')))
        self.fake_files = sorted(glob(os.path.join(fake_dir, '*.png')))

        print('real samples: ', len(self.real_files))
        print('fake samples: ', len(self.fake_files))

        self.real_dir = real_dir

    def __len__(self) : 
        return min(len(self.fake_files), 50000)
    
    def __getitem__(self, idx) : 
        fn_fake = self.fake_files[idx]
        fn_real = self.real_files[idx]

        real_image = Image.open(fn_real)
        # make the resolution of real images same as the fake images.
        real_image = real_image.resize((256,256), Image.BILINEAR)
        real_image = real_image.resize((299,299), Image.BILINEAR)
        real_image = T.ToTensor()(real_image) * 255.0
        real_image = real_image.to(torch.uint8)

        fake_image = Image.open(fn_fake)
        fake_image = fake_image.resize((299,299), Image.BILINEAR)
        fake_image = T.ToTensor()(fake_image) * 255.0
        fake_image = fake_image.to(torch.uint8)

        ret = {
            'fn_real': fn_real,
            'fn_fake': fn_fake,
            'real_image': real_image,
            'fake_image': fake_image
        }
        return ret

dataset = MetricDataset(
    real_dir=ROOT_REAL,
    fake_dir=ROOT_FAKE
)

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=16)

metric = FID(features=64).cuda()

for data in tqdm(dataloader):
    real = data['real_image'].cuda()
    fake = data['fake_image'].cuda()

    metric.update(real, real=True)
    metric.update(fake, real=False)

score = metric.compute()

print('====== FID score ======')
print('score :', score.item())
print('=======================')

# print('====== KID score ======')
# print('kid mean :', kid_mean)
# print('kid std :', kid_std)
# print('=======================')
