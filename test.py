from network import U_Net,R2U_Net,AttU_Net,R2AttU_Net,PLUNet,PUNet,LUNet,PLUNet512,NestedUNet,PLUNet32
from torchsummary import summary
from torchstat import stat
import torch
from model.DenseUnet import DenseUnet
from model.UNet3Plus import UNet3Plus
from model.kiunet import kiunet
from model.MultiResUnet import MultiResUnet
from model.DoubleUNet import DoubleUNet
from model.DenseNet import DenseNet

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NestedUNet().to(device)
    model2 = DoubleUNet().to('cpu')
    # summary(model, input_size=(3,96,96))
    # stat(model2, (3,96,96))
    # stat(model2, (3,224,224))
    # stat(model2, (3,256,256))
    stat(model2, (3,288,384))


