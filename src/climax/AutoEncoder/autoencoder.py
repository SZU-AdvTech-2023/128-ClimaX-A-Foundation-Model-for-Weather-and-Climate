import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, c) -> None:
        '''
        c -> number of encoder output channel
        '''
        super(AE, self).__init__()
        
        self.Encoder = nn.Sequential(
            nn.Conv2d(512, 256, 5, 2, 2), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, 5, 2, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, c, 3, 1, 1),
            nn.BatchNorm2d(c),
            nn.LeakyReLU()
        )
        self.Decoder = nn.Sequential(
            nn.ConvTranspose2d(c, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 256, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 512, 5, 2, 2, output_padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU()
        )

    def forward(self,x):
        x = self.Encoder(x)
        x = self.Decoder(x)
        return x 

    def encode(self, x):
        return self.Encoder(x)

    def decode(self, x):
        return self.Decoder(x)
    
if __name__ == "__main__":
    from torchsummary import summary
    input_size = (512, 512, 1024)
    print(input_size)
    summary(AE(3).to("cuda"), input_size, batch_size=-1, device="cuda")
    