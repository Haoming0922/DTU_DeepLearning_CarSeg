import os
from torch import nn,optim
import torch
from torch.utils.data import DataLoader
from dataloader import *
from net import *
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
from iouEval import iouEval
from unetpp import *
from utils.losses import hybrid_loss, CriterionStructuralKD, CrossEntropyLoss2d
from tqdm import tqdm
from sklearn.model_selection import KFold

epochs = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
kfold = KFold(n_splits=5, shuffle=True)
dataset = MyDataset("/dtu/blackhole/11/173553/carseg_data/train")
writer = SummaryWriter() 

train_loss,val_loss = [],[]
for fold,(train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print('Fold {}'.format(fold + 1))
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
    trainloader = DataLoader(dataset,batch_size=32, shuffle=False,sampler=train_subsampler)
    valloader = DataLoader(dataset,batch_size=32, shuffle=False,sampler=val_subsampler)

    # net = UNet(3, 9).to(device)
    net = NestedUNet(3, 9).to(device)

    opt = optim.Adam(net.parameters(),lr=0.002)
    
    loss_fun = CrossEntropyLoss2d()
    # loss_fun = hybrid_loss()
    
    for epoch in tqdm(range(epochs)):
        net.train()
        losses = []
        for i, (x,y,_) in enumerate(trainloader):
            x,y = x.to(device), y.to(device)
            out = net.float()(x.float())  #out_image.shape = [2, 3, 256, 256]
            y = np.squeeze(y, axis=1)
            loss = loss_fun(out[0], y.long())
            opt.zero_grad()
            loss.backward()
            opt.step()    
            losses.append(loss.data.item())
        train_loss.append(np.mean(losses))y
        writer.add_scalar('data/trainloss', np.mean(losses), epoch)    

        net.eval()
        iouEvalVal = iouEval(9,10)
        losses = []
        for i, (x,y,_) in enumerate(valloader):
            x,y = x.to(device), y.to(device)
            out = net.float()(x.float())  #out_image.shape = [2, 3, 256, 256]
            iouEvalVal.addBatch(out[0].max(1)[1].unsqueeze(1).data, segment_image.long().data)
            y = np.squeeze(y, axis=1)
            loss = loss_fun(out[0], y.long())  
            losses.append(loss.data.item())
        val_loss.append(np.mean(losses))
        writer.add_scalar('data/valloss', np.mean(losses), epoch) 
        print("train loss: ",train_loss[fold*epochs+epoch],", val loss: ",val_loss[fold*epochs+epoch],'\n')
    
        iouVal, iou_classes = iouEvalVal.getIoU()
        print("*"*50)
        print("iou: ",iouVal, "\niou class: ",iou_classes,'\n')
        
    print("Fold",fold+1,"---------train loss: ",train_loss[(fold+1)*epochs-1],", val loss: ",val_loss[(fold+1)*epochs-1],'\n')
    torch.save(net.state_dict(), 'checkpoints/model_fold_{}.pth'.format(fold+1))


writer.close()