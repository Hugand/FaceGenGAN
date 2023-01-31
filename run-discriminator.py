from discriminator import Discriminator
import numpy as np
import torch 
torch.manual_seed(17)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

disc = Discriminator()
img = torch.from_numpy(np.random.normal(size=(32, 32)).astype(np.float32))
img = img.unsqueeze(0)
img = img.unsqueeze(0)

print(img.shape)

probs = disc(img)

print("Output:", probs.detach().numpy())

