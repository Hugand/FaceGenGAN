import torch
import torchvision
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FaceGenGAN(nn.Module):
    def __init__(self, generator_epochs=10, discriminator_epochs=10, **kwargs):
        super().__init__()
        self.GENERATOR_EPOCHS = generator_epochs
        self.DISCRIMINATOR_EPOCHS = discriminator_epochs

        #self.input_layer = nn.Linear(
        #    in_features=kwargs["input_shape"], out_features=128
        #)
        self.conv1_layer = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Dropout(p=1 - 0.8))
        self.conv2_layer = nn.Sequential(
            nn.Conv2d(16, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(p=0.3),
            )#nn.ReLU(),)
        
        self.output_layer = nn.Linear(
            in_features=32*32*8, out_features=kwargs["output_shape"]
        )

    def forward(self, features):
        out = self.conv1_layer(features)
        out_relu = torch.relu(out)

        residual = out
        out = self.conv4_layer(out_relu)
        out_relu = torch.relu(out) + residual

        out = self.conv6_layer(out_relu)
        out_relu = torch.relu(out)

        flattened = out.view(out_relu.size(0), -1)

        out = self.output_layer(flattened)
        reconstructed = torch.relu(out)
        return reconstructed.view(-1, 1, 32, 32)

    def train(self, train_data, valid_data, optimizer, criterion=None, epochs=10, batch_size=64):
        losses = []
        valid_losses = []
        valid_features_small = Variable(valid_data[0][0].view(-1).to(device)).view(-1, 1, 32, 32).to(device)
        valid_features = Variable(valid_data[0][1].view(-1).to(device)).view(-1, 1, 32, 32).to(device)

        if criterion == None:
          criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            loss = 0
            print("Epoch {}/{} - ".format(epoch, epochs), end="")
            n_batches = 0
            print_step = len(train_data) * 0.1
            
            for batch_features_small, batch_features in train_data:
                # reshape mini-batch data to [N, 784] matrix
                # load it to the active device
                batch_features_small = Variable(batch_features_small.view(-1).to(device)).view(-1, 1, 32, 32).to(device)
                batch_features = Variable(batch_features.view(-1, 1, 32, 32).to(device))
                
                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()
                
                # compute reconstructions
                outputs = self(batch_features_small.float())
                
                # compute training reconstruction loss
                train_loss = criterion(outputs, batch_features.float())
                
                # compute accumulated gradients
                train_loss.backward()
                
                # perform parameter update based on current gradients
                optimizer.step()
                
                # add the mini-batch training loss to epoch loss
                loss += train_loss.item()

                #progress += step_size
                n_batches += 1

                if n_batches >= print_step:
                  #print(progress, step_size, print_step,len(train_data), n_batches)
                  print("#", end="")
                  n_batches = 0
                  
            #torch.cuda.empty_cache()
            if epoch % 5 == 0:
              valid_outputs = self(valid_features_small.float())
              valid_loss = criterion(valid_outputs, valid_features.float())
              valid_losses.append(valid_loss)
            else:
              valid_losses.append(valid_losses[-1])
            
            # compute the epoch training loss
            loss = loss / len(train_data)
            losses.append(loss)
            
            # display the epoch training loss
            vl = []

            for i in valid_losses:
              vl.append(torch.Tensor.cpu(i).detach().numpy())
            vl
            clear_output(wait=True)

            print(valid_losses)

            # Display stuff
            f = plt.figure()
            f.set_figwidth(18)
            f.set_figheight(14)
            plt.title("Train loss")
            plt.plot(losses, color="orange")
            plt.plot(vl, color="blue")
            plt.show()
            print("loss = {:.6f}, valid_loss = {:.6f}".format(loss, valid_loss))
            #plt.clf()

        return losses, valid_losses