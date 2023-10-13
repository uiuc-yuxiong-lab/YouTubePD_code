import torch
torch.backends.cudnn.benchmark = True
from torchvision import transforms, utils
from util import *
from PIL import Image
import math
import random
import os

import numpy as np
from torch import nn, autograd, optim
from torch.nn import functional as F
from tqdm import tqdm
import wandb
from model import *
from e4e_projection import projection as e4e_projection


from copy import deepcopy


os.makedirs('inversion_codes', exist_ok=True)
os.makedirs('style_images', exist_ok=True)
os.makedirs('style_images_aligned', exist_ok=True)
os.makedirs('models', exist_ok=True)

latent_dim = 512
device = 'cuda' 
# Load original generator
original_generator = Generator(1024, latent_dim, 8, 2).to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
original_generator.load_state_dict(ckpt["g_ema"], strict=False)
mean_latent = original_generator.mean_latent(10000)

# to be finetuned generator
generator = deepcopy(original_generator)


transform = transforms.Compose(
    [
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
plt.rcParams['figure.dpi'] = 150

#@title Choose input face
#@markdown Add your own image to the test_input directory and put the name here
filename = 'iu.jpeg' #@param {type:"string"}
filepath = f'test_input/{filename}'

# uploaded = files.upload()
# filepath = list(uploaded.keys())[0]
name = strip_path_extension(filepath)+'.pt'

# aligns and crops face
aligned_face = align_face(filepath)

# my_w = restyle_projection(aligned_face, name, device, n_iters=1).unsqueeze(0)
my_w = e4e_projection(aligned_face, name, device).unsqueeze(0)

display_image(aligned_face, title='Aligned face')
plt.savefig("mygraph.png")

#@markdown Upload your own style images into the style_images folder and type it into the field in the following format without the directory name. Upload multiple style images to do multi-shot image translation
# names = ['sketch.jpeg', 'sketch2.jpeg', 'sketch3.jpeg', 'sketch4.jpeg'] #@param {type:"raw"}
data_dir = 'PD_GEN_TRAIN/pos'
names = os.listdir(data_dir)

targets = []
latents = []
new_path = []
for idx,name in tqdm(enumerate(names)):
    style_path = os.path.join(data_dir, name)
    assert os.path.exists(style_path), f"{style_path} does not exist!"

    name = strip_path_extension(name)

    # crop and align the face
    style_aligned_path = os.path.join('aligned', f'{name}.png')
    if not os.path.exists(style_aligned_path):
        try:
            style_aligned = align_face(style_path)
            style_aligned.save(style_aligned_path)
            new_path.append(names[idx])
        except:
            # names.remove(name)
            continue
    else:
        style_aligned = Image.open(style_aligned_path).convert('RGB')
        new_path.append(names[idx])

    # GAN invert
    style_code_path = os.path.join('new_inv', f'{name}.pt')
    if not os.path.exists(style_code_path):
        latent = e4e_projection(style_aligned, style_code_path, device)
    else:
        latent = torch.load(style_code_path)['latent']

    targets.append(transform(style_aligned).to(device))
    latents.append(latent.to(device))
names = new_path
ori_targets = torch.stack(targets, 0)
ori_latents = torch.stack(latents, 0)

# target_im = utils.make_grid(targets, normalize=True, range=(-1, 1))
# plt.savefig('style_reference.png')

alpha =  1.0 #@param {type:"slider", min:0, max:1, step:0.1}
alpha = 1-alpha

#@markdown Tries to preserve color of original image by limiting family of allowable transformations. Set to false if you want to transfer color from reference image. This also leads to heavier stylization
preserve_color = False #@param{type:"boolean"}
#@markdown Number of finetuning steps. Different style reference may require different iterations. Try 200~500 iterations.
num_iter = 3000 #@param {type:"number"}
#@markdown Log training on wandb and interval for image logging
use_wandb = False #@param {type:"boolean"}
log_interval = 50 #@param {type:"number"}


# load discriminator for perceptual loss
discriminator = Discriminator(1024, 2).eval().to(device)
ckpt = torch.load('models/stylegan2-ffhq-config-f.pt', map_location=lambda storage, loc: storage)
discriminator.load_state_dict(ckpt["d"], strict=False)

# reset generator
del generator
generator = deepcopy(original_generator)

g_optim = optim.Adam(generator.parameters(), lr=2e-3, betas=(0, 0.99))

# Which layers to swap for generating a family of plausible real images -> fake image
if preserve_color:
    id_swap = [9,11,15,16,17]
else:
    id_swap = list(range(7, generator.n_latent))


if not os.path.exists('pd.pth'):
    for idx in tqdm(range(num_iter)):
        # rand_idx = torch.randint([0, len(ori_targets), (4,)])
        rand_idx = torch.randint(0, len(ori_targets), (3,))
        latents = ori_latents[rand_idx]
        targets = ori_targets[rand_idx]
        mean_w = generator.get_latent(torch.randn([latents.size(0), latent_dim]).to(device)).unsqueeze(1).repeat(1, generator.n_latent, 1)
        in_latent = latents.clone()
        in_latent[:, id_swap] = alpha*latents[:, id_swap] + (1-alpha)*mean_w[:, id_swap]

        img = generator(in_latent, input_is_latent=True)

        with torch.no_grad():
            real_feat = discriminator(targets)
        fake_feat = discriminator(img)

        loss = sum([F.l1_loss(a, b) for a, b in zip(fake_feat, real_feat)])/len(fake_feat)
        

        g_optim.zero_grad()
        loss.backward()
        g_optim.step()
        del latents,targets, mean_w, in_latent, img, real_feat, fake_feat, loss
        torch.cuda.empty_cache()
    torch.save(generator.state_dict(), 'pd.pth')
else:
    generator.load_state_dict(torch.load('pd.pth'))

    
#@title Generate results
n_sample =  5#@param {type:"number"}
seed = 3000 #@param {type:"number"}

torch.manual_seed(seed)
with torch.no_grad():
    generator.eval()
    z = torch.randn(n_sample, latent_dim, device=device)

    original_sample = original_generator([z], truncation=0.7, truncation_latent=mean_latent)
    sample = generator([z], truncation=0.7, truncation_latent=mean_latent)


style_images = []
for name in names:
    style_path = f'aligned/{strip_path_extension(name)}.png'
    style_image = transform(Image.open(style_path))
    style_images.append(style_image)
    

output = torch.cat([original_sample, sample], 0)
display_image(utils.make_grid(output, normalize=True, range=(-1, 1), nrow=n_sample), title='Random samples')
plt.savefig('random.png')
data_dir = 'PD_GEN_TEST/neg'
images = os.listdir(data_dir)
for im in images:
    filepath = os.path.join(data_dir, im)
    try:
        aligned_face = align_face(filepath)
        aligned_face.save('aligned_new/'+im)
    except:
        aligned_face = Image.open(filepath).convert('RGB')
    my_w = e4e_projection(aligned_face, 'inversion_codes/'+im, device).unsqueeze(0)
    my_sample = generator(my_w, input_is_latent=True)
    img = my_sample[0].detach().cpu().squeeze().permute(1,2,0).numpy()
    Image.fromarray(((img+1)/2*255).astype(np.uint8)).save('jojo_test_out_new/'+im)