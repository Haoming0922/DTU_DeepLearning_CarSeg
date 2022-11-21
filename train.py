import os

from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from data import *
from net import *
from torchvision.utils import save_image
from tensorboardX import SummaryWriter


writer = SummaryWriter() #可视化
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weight_path='params/unet.pth'
data_path=r'data'
save_path='train_image'
if __name__ == '__main__':
    data_loader=DataLoader(MyDataset(data_path),batch_size=2,shuffle=True)
    net=UNet().to(device)
    if os.path.exists(weight_path):
        net.load_state_dict(torch.load(weight_path))
        print('successful load weight！')
    else:
        print('not successful load weight')

    opt=optim.Adam(net.parameters())
    loss_fun=nn.BCELoss()

    epoch=1

    while True:
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch, 10000))
        for i,(image,segment_image) in enumerate(data_loader):
            image, segment_image=image.to(device),segment_image.to(device)
            # print(torch.unique(segment_image))
            # print('type(segment_image):', type(segment_image), 
            # 'segment_image.shape: ', segment_image.shape, 'image.shape:', image.shape)    image.shape = [2, 3, 256, 256]  segment.shape = [2, 3, 256, 256]
            out_image=net(image) # out_image.shape = [2, 3, 256, 256]
            train_loss=loss_fun(out_image,segment_image)

            opt.zero_grad()
            train_loss.backward()
            opt.step()
            running_loss += train_loss.data.item()
            epoch_loss = running_loss / epoch

            if i%5==0:
                print(f'{epoch}-{i}-train_loss===>>{train_loss.item()}')

            if i%100==0:
                torch.save(net.state_dict(),weight_path)

                _image=image[0]
                _segment_image=segment_image[0]
                _out_image=out_image[0]
                # print("++++++++++++++out_image:", _out_image)

                img=torch.stack([_image,_segment_image,_out_image],dim=0)
                save_image(img,f'{save_path}/{i}.png')

            writer.add_scalar('data/trainloss', epoch_loss, epoch)

        if epoch%1000 == 0:
            torch.save(net, 'checkpoints/model_epoch_{}.pth'.format(epoch))
            print('checkpoints/model_epoch_{}.pth saved!'.format(epoch))

        epoch+=1

writer.close()
