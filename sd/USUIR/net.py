import torch
from sd.USUIR.ITA import JNet, TNet

class net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.image_net = JNet()
        self.mask_net = TNet()

        # self.dec = ColorNet()
        # self.mlp = MLP(6, get_num_adain_params(self.dec), 64, 3, norm='none', activ='relu')


    def forward(self, data):
        x_j = self.image_net(data)
        x_t = self.mask_net(data)
        return x_j, x_t





