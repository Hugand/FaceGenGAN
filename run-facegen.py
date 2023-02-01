from facegen_gan import FaceGenGAN
import numpy as np
import matplotlib.pyplot as plt
import torch 
from torch import nn
import pickle
torch.manual_seed(17)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

with open('faces_imgs.pkl','rb') as f:
    real_imgs = pickle.load(f)[:256].reshape((256, 1, 32, 32)).astype(np.float32)
    print(real_imgs.shape)

    facegen = FaceGenGAN()

    # optimizer = torch.optim.Adam(facegen.generator.parameters(), lr=0.002)
    # loss_criterion = nn.CrossEntropyLoss()

    facegen.train(torch.from_numpy(real_imgs), epochs= 5, batch_size=32)
