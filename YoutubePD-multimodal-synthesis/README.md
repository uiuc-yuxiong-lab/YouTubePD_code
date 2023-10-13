

# YoutubePD Multimodal & Progression Synthesis
In this repo, we provide the multimodal and synthesis benchmark for YoutubePD dataset. We also provide the code for baselines.
## Data Preparation

### Preprocessed data
We provide preprocessed data for image, audio and landmark parts. please download [here](https://drive.google.com/file/d/1ofeOXywHlhXlWQiwclEHGhZFKJAxHk0E/view?usp=sharing)

Please put the data dir under the root of the repo. The data dictionary structure should follow:
```
data
├── PD_DET
│   ├── train.csv
│   ├── train_set
│   │   ├── audio
│   │   ├── imgs
│   │   └── kpts_abs
│   ├── val.csv
│   └── val_set
│       ├── audio
│       ├── imgs
│       └── kpts_abs
└── PD_GEN
    ├── PD_GEN_TEST
    │   ├── neg
    │   └── pos
    └── PD_GEN_TRAIN
        ├── neg
        └── pos
```

### Manualy process the multimodal data
Alternatively, you can process the data by your self. 

#### Image Data & Landmark Data
Please use the `extract.py` to extract the image and landmark data. As we use the api from Face++, please replace the key with your own api key in the file.
#### Audio Data
Please follow our audio single modal process to extract the audio data

## Multimodal Baseline
Our method relies on the VGG face pretrained model. The weights can be downloaded [here](https://drive.google.com/file/d/1Zq5b9h-qlEVvK_aZKIzy8Y29Xq7bylqb/view?usp=sharing). Please put the checkpoint under `./multimodal/`

We provide FV based multimodal baseline. Please run the following command to train the model:
```
cd multimodal
python train.py
```
## PD Progression Baselines
We build baselines from various previous methods. Please following their official repo for experiments. 
### Training

#### We provide suggestions for each method.
For StyleGAN2, few-shot adaptation and JoJoGAN, please  first follow the steps in JoJoGAN to align the images. Then, use `/PD_GEN_TRAIN/pos` for GAN training. During inference, JoJoGAN provide model to create latent code for generation. For StyleGAN2 and few-shot adaptation, we use the projector.py to get correspond latent code. Please make sure the output image maintain their original names. Example code are provided in `synthesis/GAN_proj/projector.py` and `synthesis/JoJoGAN/stylize.py`. Please place them under orignal repo and modify the path in the code.
* StyleGAN2: https://github.com/rosinality/stylegan2-pytorch
* Few-shot adaptation: https://github.com/utkarshojha/few-shot-gan-adaptation
* JoJoGAN: https://github.com/mchong6/JoJoGAN

For CUT and CycleGAN, please use `/PD_GEN_TRAIN/neg` as the source domain and `/PD_GEN_TRAIN/pos` as the target domain. Please follow the official repo for training and inference.
* CUT: https://github.com/taesungp/contrastive-unpaired-translation
* CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

For HRFAE, we provide a pretrained binary classification model [here](https://drive.google.com/file/d/1OZ9neQgRNQcgftEXEIzGtECNgITwF101/view?usp=sharing). Please follow the code `synthesis/HRFAE/model_test.py` to load it and replace the original classifier with this.
* HRFAE: https://github.com/InterDigitalInc/HRFAE

For StableDiffusion with LoRA finetuning, please use `/PD_GEN_TRAIN/pos` for finetuning. Please follow the official repo's img2img mode for inference.
* LoRA: https://github.com/cloneofsimo/lora

### Testing

Please install [pytorch-fid](https://github.com/mseitzer/pytorch-fid)
To evaluate, please translate the `PD_GEN/PD_GEN_TEST/neg` images to positive images.

You can simply evaluate by running the following command:
```
cd synthesis/metrics
bash test_all.sh /path/to/generated/images \
 ./data/PD_GEN/PD_GEN_TEST/neg \
 ./data/PD_GEN/PD_GEN_TEST/pos
```







