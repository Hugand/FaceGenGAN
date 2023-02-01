from discriminator import Discriminator
import numpy as np
import torch 
torch.manual_seed(17)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

disc = Discriminator()
img = torch.from_numpy(np.random.normal(size=(3, 1, 32, 32)).astype(np.float32))

print(img.shape)

probs = disc(img)

print("Output:", probs.detach().numpy())

