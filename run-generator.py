from generator import Generator
import numpy as np
import matplotlib.pyplot as plt
import torch 
torch.manual_seed(17)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

gen = Generator()
img = torch.from_numpy(np.random.normal(size=(32, 32)).astype(np.float32))
img = img.unsqueeze(0)
img = img.unsqueeze(0)

print(img.shape)

generated_img = gen(img)

plt.rcParams["figure.figsize"] = [15, 9]
fig, axs = plt.subplots(2)

#for ax in axs:
axs[0].imshow(img[0][0], cmap='inferno')
axs[1].imshow(generated_img.detach().numpy()[0][0], cmap='inferno')
plt.show()