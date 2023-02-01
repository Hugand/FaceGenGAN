import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
import numpy as np
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FaceGenGAN(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		self.generator = Generator().to(device)
		self.discriminator = Discriminator().to(device)

	def forward(self, features):
		return self.generator(features)

	def train_generator(self, n_batches, batch_size, optimizer, loss_criterion):
		loss = 0
		batch_progress = 0
		print_step = n_batches * 0.1

		for i in range(n_batches):
			# Generate batch noisy images
			noise_imgs = torch.from_numpy(np.random.random_sample(size=(batch_size, 1, 32, 32)).astype(np.float32)).to(device)
			noise_imgs = Variable(noise_imgs, requires_grad=True)
			# reset the gradients back to zero
			# PyTorch accumulates gradients on subsequent backward passes
			optimizer.zero_grad()
			
			# compute reconstructions
			outputs = self.discriminator(self.generator(noise_imgs))
			
			# compute training reconstruction loss
			train_loss = loss_criterion(outputs, torch.from_numpy(np.array([[1]] * batch_size, dtype=np.float32)).to(device))

			# compute accumulated gradients for generator and discriminator
			train_loss.backward()
			
			# perform parameter update based on current gradients
			# only for the generator
			optimizer.step()

			#print("2", list(self.generator.parameters())[0].grad[:2])
			
			# add the mini-batch training loss to epoch loss
			loss += train_loss.item()

			#progress += step_size
			batch_progress += 1

			if batch_progress >= print_step:
				#print(progress, step_size, print_step,len(train_data), n_batches)
				print("#", end="")
				batch_progress = 0

		return loss / n_batches

	def train_discriminator(self, real_imgs, n_batches, batch_size, optimizer, loss_criterion):
		loss = 0
		batch_progress = 0
		print_step = n_batches * 0.1

		for i in range(n_batches):
			# Generate batch noisy images
			noise_imgs = torch.from_numpy(np.random.random_sample(size=(int(batch_size / 2), 1, 32, 32)).astype(np.float32)).to(device)
			noise_imgs = self.generator(noise_imgs)
			batch_imgs = torch.cat([real_imgs[i], noise_imgs], dim=0).to(device)
			batch_imgs_labels = torch.from_numpy(np.array( ([[1]] * int(batch_size / 2)) + ([[0]] * int(batch_size / 2)), dtype=np.float32 )).to(device)
			batch_imgs = Variable(batch_imgs, requires_grad=True)
			batch_imgs_labels = Variable(batch_imgs_labels, requires_grad=True)
			#print(noise_imgs[0][0][0])

			# reset the gradients back to zero
			# PyTorch accumulates gradients on subsequent backward passes
			optimizer.zero_grad()
			
			# compute reconstructions
			outputs = self.discriminator(batch_imgs.to(device))
			
			# compute training reconstruction loss
			train_loss = loss_criterion(outputs, batch_imgs_labels)

			# compute accumulated gradients for generator and discriminator
			train_loss.backward()
			
			# perform parameter update based on current gradients
			# only for the generator
			optimizer.step()
			#print("2", list(self.discriminator.parameters())[0].grad[:2])
			
			# add the mini-batch training loss to epoch loss
			loss += train_loss.item()

			#progress += step_size
			batch_progress += 1

			if batch_progress >= print_step:
				#print(progress, step_size, print_step,len(train_data), n_batches)
				print("#", end="")
				batch_progress = 0

		return loss / n_batches * 2
		

	def train(self, real_imgs, epochs=10, batch_size=64, generator_epochs=10, discriminator_epochs=10):
		losses = []
		valid_losses = []
		# valid_features_small = Variable(valid_data[0][0].view(-1).to(device)).view(-1, 1, 32, 32).to(device)
		# valid_features = Variable(valid_data[0][1].view(-1).to(device)).view(-1, 1, 32, 32).to(device)
		N_BATCHES = int(len(real_imgs) / batch_size)
		training_phases = [('DISCRIMINATOR', discriminator_epochs), ('GENERATOR', generator_epochs)]
		training_phase = 0
		training_phase_count = 0
		real_imgs = torch.reshape(real_imgs, (N_BATCHES*2, int(batch_size / 2)) + real_imgs.shape[1:]).to(device)
		print(f'Real imgs shape: {real_imgs.shape} - {device}')

		criterion = nn.BCEWithLogitsLoss()

		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=0.1, betas=(0.5, 0.999))
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=0.1, betas=(0.5, 0.999))
		print("Start training...")
		for epoch in range(epochs):
			loss = 0
			print(f'[{training_phases[training_phase]}] Epoch {epoch}/{epochs} - ', end="")

			if training_phase == 0:
				loss = self.train_discriminator(real_imgs, N_BATCHES, batch_size, discriminator_optimizer, criterion)
			else:
				loss = self.train_generator(N_BATCHES, batch_size, generator_optimizer, criterion)

			losses.append(loss)

			print(f', loss: {loss}')

			#torch.cuda.empty_cache()
			# if epoch % 5 == 0:
			# 	valid_outputs = self(valid_features_small.float())
			# 	valid_loss = criterion(valid_outputs, valid_features.float())
			# 	valid_losses.append(valid_loss)
			# else:
			# 	valid_losses.append(valid_losses[-1])
			
			# compute the epoch training loss

			training_phase_count += 1

			if training_phase_count >= training_phases[training_phase][1]:
				training_phase = int(not training_phase)
				training_phase_count = 0
			
			# # display the epoch training loss
			# vl = []

			# for i in valid_losses:
			# 	vl.append(torch.Tensor.cpu(i).detach().numpy())

			# print(valid_losses)

			# # Display stuff
			# f = plt.figure()
			# f.set_figwidth(18)
			# f.set_figheight(14)
			# plt.title("Train loss")
			# plt.plot(losses, color="orange")
			# plt.plot(vl, color="blue")
			# plt.show()
			# print("loss = {:.6f}, valid_loss = {:.6f}".format(loss, valid_loss))
			#plt.clf()

		return losses, valid_losses