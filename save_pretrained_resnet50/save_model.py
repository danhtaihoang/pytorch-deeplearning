import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os,sys,time,platform
import openslide
from PIL import Image

import torch
#from torch import nn
import torchvision.transforms as transforms
from torchvision.models import resnet50

from utils_preprocessing import *

#import utils_color_norm
#color_norm = utils_color_norm.macenko_normalizer()

## check available device
#device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
#print("device:", device)
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

init_random_seed(random_seed=42)
##======================================================================================================
start_time = time.time()

##======================================================================================================
path2tiles = "test_10tiles/"
## collect selecting tile names within a slide folder
tile_names = []
for f in os.listdir(path2tiles):
    if f.startswith("tile_"):
        tile_names.append(f)

## alphabet sort
tile_names = np.array(sorted(tile_names))
#print(tile_names)

n_tiles = len(tile_names)
print("n_tiles:", n_tiles)

##======================================================================================================
model = Feature_Extraction(model_type="load_from_internet")
model.eval()

data_transform = transforms.Compose([transforms.Resize(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                          std=[0.229, 0.224, 0.225])])
##======================================================================================================
tiles_list = []
for i_tile in range(n_tiles):
    tile_name = tile_names[i_tile]
    tile = Image.open(f"{path2tiles}{tile_name}").convert('RGB')
    tiles_list.append(tile)

n_tiles_selected = len(tiles_list)
##---------------------------------------
tiles = []
for i in range(n_tiles_selected):
    tiles.append(data_transform(tiles_list[i]).unsqueeze(0))
tiles = torch.cat(tiles, dim=0)
print("tiles.shape:", tiles.shape)
tiles_list = 0


np.save("tiles.npy", tiles)

##---------------------------------------
batch_size = 64
features = []
for idx_start in range(0, n_tiles_selected, batch_size):
    idx_end = idx_start + min(batch_size, n_tiles_selected - idx_start)

    feature = model(tiles[idx_start:idx_end])
    
    features.append(feature.detach().cpu().numpy())

features = np.concatenate(features)

print("features.shape:", features.shape)
np.save("features_saved.npy", features)
##======================================================================================================
##======================================================================================================
## save the model
#torch.save(model,"ResNet50_IMAGENET1K_V2.pt")
torch.save(model.state_dict(), "ResNet50_IMAGENET1K_V2.pt", _use_new_zipfile_serialization=False)


print("time:", (time.time() - start_time))
print("--- finished --- ")


