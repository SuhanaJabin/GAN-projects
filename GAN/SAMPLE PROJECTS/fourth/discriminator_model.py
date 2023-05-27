import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self,in_channels, out_channels, stride):
        super().__init__() #gets all the functionalities of the nn.Module to the block
        #creating a conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,4, stride,1, bias=True, padding_mode="reflect"),
            #4 is the kernel size and 1 is padding
            #using reflect reduces the artifact(distortions in image)
            nn.InstanceNorm2d(out_channels), #Normalization is done to reduce the value to 0 or 1
            #Normalization - why needed- for restricting the activations within a small range.
            nn.LeakyReLU(0.2),#done to introduce non-linearity
    
        )

    def forward(self,x):
        return self.conv(x)
    
class Discriminator(nn.Module):
    def __init__(self,in_channels=3,features=[64,128,256,512]):