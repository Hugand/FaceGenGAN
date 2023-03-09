import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from generator import Generator
from discriminator import Discriminator
from IPython.display import clear_output
import numpy as np
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FaceGenGAN(nn.Module):

	def __init__(self, **kwargs):
		super().__init__()
		self.generator = Generator().to(device)
		self.generator.apply(self.__weights_init)
		self.discriminator = Discriminator().to(device)
		self.discriminator.apply(self.__weights_init)
		

	def __weights_init(self, m):
		classname = m.__class__.__name__
		if classname.find('Conv') != -1:
			torch.nn.init.normal_(m.weight, 0.0, 0.02)
		elif classname.find('BatchNorm') != -1:
			torch.nn.init.normal_(m.weight, 1.0, 0.02)
			torch.nn.init.zeros_(m.bias)

	def forward(self, features):
		return self.generator(features)

	def train_generator(self, n_batches, batch_size, optimizer, loss_criterion):
		loss = 0
		batch_progress = 0
		print_step = n_batches * 0.1

		for i in range(n_batches):
			# Generate batch noisy images
			#noise_imgs = torch.from_numpy(np.random.random_sample(size=(batch_size, 1, 32, 32)).astype(np.float32)).to(device)
			noise_imgs = torch.rand(batch_size, 32*32, 1, 1, device=device)
			#noise_imgs = Variable(noise_imgs, requires_grad=True)
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
		print_step = n_batches * 0.
		train_acc = 0.0

		for i in range(n_batches):
			# Generate batch noisy images
			#noise_imgs = torch.from_numpy(np.random.random_sample(size=(int(batch_size / 2), 1, 32, 32)).astype(np.float32)).to(device)
			
			#print(noise_imgs.shape)
			#batch_imgs = torch.cat([real_imgs[i], noise_imgs], dim=0).to(device)
			#batch_imgs_labels = torch.from_numpy(np.array( ([[1]] * int(batch_size / 2)) + ([[0]] * int(batch_size / 2)), dtype=np.float32 )).to(device)
			#batch_imgs = Variable(batch_imgs, requires_grad=True)
			#batch_imgs_labels = Variable(batch_imgs_labels, requires_grad=True)
			#print(noise_imgs[0][0][0])

			# reset the gradients back to zero
			# PyTorch accumulates gradients on subsequent backward passes
			optimizer.zero_grad()
			
			real_labels = torch.full((int(batch_size / 2),), 1, dtype=torch.float, device=device)
			# compute reconstructions
			outputs = self.discriminator(real_imgs[i].to(device)).view(-1)
			# compute training reconstruction loss
			d_real_loss = loss_criterion(outputs, real_labels)
			# compute accumulated gradients for generator and discriminator
			d_real_loss.backward()
			d_loss = d_real_loss.mean().item()
			train_acc += torch.sum(outputs == 1)
			
			noise_imgs = torch.rand(int(batch_size / 2), 32*32, 1, 1, device=device)
			noise_imgs = self.generator(noise_imgs)
			fake_labels = torch.full((int(batch_size / 2),), 0, dtype=torch.float, device=device)
			# compute reconstructions
			outputs = self.discriminator(noise_imgs.detach()).view(-1)
			# compute training reconstruction loss
			d_fake_loss = loss_criterion(outputs, fake_labels)
			# compute accumulated gradients for generator and discriminator
			d_fake_loss.backward()
			g_loss = d_fake_loss.mean().item()
			train_acc += torch.sum(outputs == 0)

			d_loss = d_real_loss + d_fake_loss

			# perform parameter update based on current gradients
			# only for the generator
			optimizer.step()
			#print("2", list(self.discriminator.parameters())[0].grad[:2])
			
			# add the mini-batch training loss to epoch loss
			loss += d_loss.item()

			#progress += step_size
			batch_progress += 1

			if batch_progress >= print_step:
				#print(progress, step_size, print_step,len(train_data), n_batches)
				print("#", end="")
				batch_progress = 0

		train_acc /= (n_batches * batch_size)
		print(", Acc: ", train_acc)

		return loss / n_batches * 2
		

	def train(self, real_imgs, epochs=10, batch_size=64, generator_epochs=10, discriminator_epochs=10, generator_lr=0.0002, discriminator_lr=0.0002):
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

		criterion_disriminator = nn.BCELoss()
		criterion_generator = nn.BCELoss()

		generator_optimizer = torch.optim.Adam(self.generator.parameters(), lr=generator_lr, betas=(0.5, 0.999))
		discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=discriminator_lr, betas=(0.5, 0.999))

		dummy_img = torch.rand(1, 32*32, 1, 1, device=device)

		print("Start training...")
		for epoch in range(epochs):
			loss = 0
			print(f'[{training_phases[training_phase]}] Epoch {epoch}/{epochs} - ', end="")

			if training_phase == 0:
				loss = self.train_discriminator(real_imgs, N_BATCHES, batch_size, discriminator_optimizer, criterion_disriminator)
			else:
				loss = self.train_generator(N_BATCHES, batch_size, generator_optimizer, criterion_generator)

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

			#clear_output(wait=True)
			with torch.no_grad():
			# Display stuff
				fig, axs = plt.subplots(1, 2)
				fig.set_figwidth(15)
				fig.set_figheight(7)

				#for ax in axs:
				axs[0].imshow(torch.Tensor.cpu(dummy_img.reshape((1, 1, 32, 32)))[0][0], cmap='inferno')
				axs[1].imshow(torch.Tensor.cpu(self.generator(dummy_img)).detach().numpy()[0][0], cmap='inferno')
				plt.show()
				#plt.clf()


		return losses, valid_losses
