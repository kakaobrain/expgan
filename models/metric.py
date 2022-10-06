# import torch
import torch.nn.functional as F
# import torchvision.models as models
# from pytorch_lightning.core.lightning import LightningModule
from torch import autograd #, nn

# from .arcface.iresnet import ArcFaceBackbone
from .StyleGAN2.op import conv2d_gradfix


# class ArcfaceLoss(LightningModule):
#     def __init__(self, path):
#         super(ArcfaceLoss, self).__init__()
#         print('Loading ResNet ArcFace')
#         self.facenet = ArcFaceBackbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
#         self.facenet.load_state_dict(torch.load(path))
#         self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
#         self.facenet.eval()

#     def extract_feats(self, x):
#         # x = x[:, :, 35:223, 32:220]  # Crop interesting region # TODO: check later
#         x = self.face_pool(x)
#         x_feats = self.facenet(x)
#         return x_feats

#     def forward(self, inp, target):
#         n_samples = inp.shape[0]
#         inp_feats = self.extract_feats(inp)  # Otherwise use the feature from there
#         target_feats = self.extract_feats(target)
#         # y_feats = y_feats.detach() # TODO: check later
#         loss = 0
#         count = 0
#         for i in range(n_samples):
#             diff_target = inp_feats[i].dot(target_feats[i])
#             loss += 1 - diff_target
#             count += 1

#         return loss / count
    
#     def train(self, mode: bool):
#         """ avoid pytorch lighting auto set trian mode """
#         return super().train(False)

#     def setup(self, device: torch.device):
#         super().setup(device)
#         self.freeze()

# # identity loss
# class ArcfaceLoss(LightningModule):
#     def __init__(self, weight, name):
#         super().__init__()
#         weight = torch.load(weight)
#         self.model = get_model(name, fp16=True)
#         self.model.load_state_dict(weight)
#         self.model.eval()
#         for param in self.model.parameters():
#             param.requires_grad = False
#         self.cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)

#     def forward(self, inp, target) :
#         inp = F.interpolate(inp, size=(112,112), mode='bilinear')
#         target = F.interpolate(target, size=(112,112), mode='bilinear')
#         inp = self.model(inp)
#         target = self.model(target)
#         loss = 1 - self.cos(inp, target)
#         return loss.mean()


# MSE = torch.nn.MSELoss()

# class GANLoss(object):
#     def __init__(self, tensor=torch.FloatTensor): 
#         self.one_tensor = None
#         self.zero_tensor = None
#         self.Tensor = tensor

#     def get_one_tensor(self, input):
#         if self.one_tensor is None: 
#             self.one_tensor = self.Tensor(1).fill_(1)
#             self.one_tensor.requires_grad_(False)
#         return self.one_tensor.expand_as(input).to(input.device)
    
#     def get_zero_tensor(self, input):
#         if self.zero_tensor is None:
#             self.zero_tensor = self.Tensor(1).fill_(0)
#             self.zero_tensor.requires_grad_(False)
#         return self.zero_tensor.expand_as(input).to(input.device)
    
#     def __call__(self, pred_fake, pred_real, for_discriminator):
#         # inputs: multiscale, multiple feature maps

#         loss = 0.0
#         for fake, real  in zip(pred_fake, pred_real):
#             _loss =  self.loss(fake, real, for_discriminator)
#             loss += _loss
                    
#         return loss / len(pred_fake)
        
        
#     def loss(self, pred_fake, pred_real, for_discriminator):
#         # single scale에 대한 loss를 계산하는 함수
#         # input: list, [features] + [final]

#         if for_discriminator : 
#             # discriminator loss term (eq.6)
#             loss = F.mse_loss(pred_fake[-1], self.get_zero_tensor(pred_fake[-1]))
#             loss += F.mse_loss(pred_real[-1], self.get_one_tensor(pred_real[-1]))
#             loss *= 0.5
#             # loss = pred_fake[-1].mean()**2
#             # loss += (1-pred_real[-1].mean())**2
#             # loss *= 0.5
                        
#         else: 
#             # adversarial generator loss term (eq.5)
#             # pred_target, pred_real is the form of list (j feature maps)
#             loss = F.mse_loss(self.get_one_tensor(pred_fake[-1]), pred_fake[-1])
#             # loss = (1-pred_fake[-1].mean())**2

#             for f_fake, f_real in zip(pred_fake[:-1], pred_real[:-1]):
#                 _loss = F.l1_loss(f_fake, f_real.detach())
#                 loss += _loss
                
#         return loss

# # Defines the GAN loss which uses either LSGAN or the regular GAN.
# # When LSGAN is used, it is basically same as MSELoss,
# # but it abstracts away the need to create the target label tensor
# # that has the same size as the input
# class GANLoss_orig(torch.nn.Module):
#     def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0,
#                  tensor=torch.FloatTensor, opt=None):
#         super(GANLoss_orig, self).__init__()
#         self.real_label = target_real_label
#         self.fake_label = target_fake_label
#         self.real_label_tensor = None
#         self.fake_label_tensor = None
#         self.zero_tensor = None
#         self.Tensor = tensor
#         self.gan_mode = gan_mode
#         self.opt = opt
#         if gan_mode == 'ls':
#             pass
#         elif gan_mode == 'original':
#             pass
#         elif gan_mode == 'w':
#             pass
#         elif gan_mode == 'hinge':
#             pass
#         else:
#             raise ValueError('Unexpected gan_mode {}'.format(gan_mode))

#     def get_target_tensor(self, input, target_is_real):
#         if target_is_real:
#             if self.real_label_tensor is None:
#                 self.real_label_tensor = self.Tensor(1).to(input.device).fill_(self.real_label)
#                 self.real_label_tensor.requires_grad_(False)
#             return self.real_label_tensor.expand_as(input)
#         else:
#             if self.fake_label_tensor is None:
#                 self.fake_label_tensor = self.Tensor(1).to(input.device).fill_(self.fake_label)
#                 self.fake_label_tensor.requires_grad_(False)
#             return self.fake_label_tensor.expand_as(input)

#     def get_zero_tensor(self, input):
#         if self.zero_tensor is None:
#             self.zero_tensor = self.Tensor(1).fill_(0)
#             self.zero_tensor.requires_grad_(False)
#         return self.zero_tensor.expand_as(input).to(input.device)

#     def loss(self, input, target_is_real, for_discriminator=True):
#         if self.gan_mode == 'original':  # cross entropy loss
#             target_tensor = self.get_target_tensor(input, target_is_real)
#             loss = F.binary_cross_entropy_with_logits(input, target_tensor)
#             return loss
#         elif self.gan_mode == 'ls':
#             target_tensor = self.get_target_tensor(input, target_is_real)
#             return F.mse_loss(input, target_tensor)
#         elif self.gan_mode == 'hinge':
#             if for_discriminator:
#                 if target_is_real:
#                     minval = torch.min(input - 1, self.get_zero_tensor(input))
#                     loss = -torch.mean(minval)
#                 else:
#                     minval = torch.min(-input - 1, self.get_zero_tensor(input))
#                     loss = -torch.mean(minval)
#             else:
#                 assert target_is_real, "The generator's hinge loss must be aiming for real"
#                 loss = -torch.mean(input)
#             return loss
#         else:
#             # wgan
#             if target_is_real:
#                 return -input.mean()
#             else:
#                 return input.mean()

#     def __call__(self, input, target_is_real, for_discriminator=True):
#         # computing loss is a bit complicated because |input| may not be
#         # a tensor, but list of tensors in case of multiscale discriminator
#         if isinstance(input, list):
#             loss = 0
#             for pred_i in input:
#                 if isinstance(pred_i, list):
#                     pred_i = pred_i[-1]
#                 loss_tensor = self.loss(pred_i, target_is_real, for_discriminator)
#                 bs = 1 if len(loss_tensor.size()) == 0 else loss_tensor.size(0)
#                 new_loss = torch.mean(loss_tensor.view(bs, -1), dim=1)
#                 loss += new_loss
#             return loss / len(input)
#         else:
#             return self.loss(input, target_is_real, for_discriminator)


# stylegan2 loss 
def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()

def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss
