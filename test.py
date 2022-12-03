import os

import torch
from torch.utils.data import DataLoader
from net import *
from dataloader import *
from torchvision.utils import save_image
from unetpp import *
from torchvision import transforms
from iouEval import iouEval
import cv2
from torch import nn
# import visdom as vis
from torch.nn import functional as F

transform_test=transforms.Compose([
    transforms.ToTensor(),
])


def eval(model, dataset_loader, criterion, num_classes):
    model.eval()
    epoch_loss_val = []
    num_cls = num_classes

    iouEvalVal = iouEval(num_cls, num_cls - 1)

    with torch.no_grad():
        for step, (images, labels, filename, filenameGt) in enumerate(dataset_loader):
            # inputs size: torch.Size([1, 20, 512, 1024])
            inputs = images.cuda()
            targets = labels.cuda()

            outputs = model(inputs)

            loss = criterion(outputs, targets[:, 0])
            epoch_loss_val.append(loss.item())

            iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)

    average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()

    #     print('check val fn, loss, acc: ', iouVal)

    return iou_classes, iouVal

def label2color(label):
    c, h, w = label.shape
    gt_color = np.zeros((3, h, w))
    print("np.unique(label)", np.unique(label))
    for i in range(h):
        for j in range(w):
            # print('label[i, j]', label[0, i, j])
            if label[0, i, j] == 0:
                gt_color[0, i, j] = 0
                gt_color[1, i, j] = 0
                gt_color[2, i, j] = 0

            if label[0, i, j] == 1:
                gt_color[0, i, j] = 0
                gt_color[1, i, j] = 128
                gt_color[2, i, j] = 0

            if label[0, i, j] == 2:
                gt_color[0, i, j] = 0
                gt_color[1, i, j] = 0
                gt_color[2, i, j] = 128

            if label[0, i, j] == 3:
                gt_color[0, i, j] = 128
                gt_color[1, i, j] = 0
                gt_color[2, i, j] = 0

            if label[0, i, j] == 4:
                gt_color[0, i, j] = 128
                gt_color[1, i, j] = 128
                gt_color[2, i, j] = 0

            if label[0, i, j] == 5:
                gt_color[0, i, j] = 128
                gt_color[1, i, j] = 0
                gt_color[2, i, j] = 128

            if label[0, i, j] == 6:
                gt_color[0, i, j] = 0
                gt_color[1, i, j] = 128
                gt_color[2, i, j] = 128

            if label[0, i, j] == 7:
                gt_color[0, i, j] = 128
                gt_color[1, i, j] = 128
                gt_color[2, i, j] = 128

            if label[0, i, j] == 8:
                gt_color[0, i, j] = 64
                gt_color[1, i, j] = 0
                gt_color[2, i, j] = 128
    return gt_color

def main():
    co_transform_val = MyCoTransform(augment=False, height=256, width=256)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data_path = '/Users/dongtianchi/Documents/DL_final/code/data/test_data'
    test_loader = DataLoader(MyDataset(data_path, transform=co_transform_val), batch_size=1, shuffle=True)

    net = UNet(3,9).cuda()
    # net = NestedUNet(3,9).cuda()
    weights = '/Users/dongtianchi/Documents/DL_final/code/checkpoints/unet++/model_epoch_40.pth'
    if os.path.exists(weights):
        net.load_state_dict(torch.load(weights))
        print('successfully')
    else:
        print('no loading')
    net.eval()

    num_cls = 9
    iouEvalVal = iouEval(num_cls, num_cls + 1)

    with torch.no_grad():
        for i, (image, segment_image, segment_name) in enumerate(test_loader):
            image, segment_image = image.to(device), segment_image.to(device)
            print(torch.unique(segment_image))
            out = net(image)
            print('---out.shape', out[0].shape)
            iouEvalVal.addBatch(out[0].max(1)[1].unsqueeze(1).data, segment_image.long().data)

            print('out', np.unique(out[0].cpu().numpy()))
            out = F.softmax(out[0], dim=1).cpu()
            print("out.shape", out.shape)
            print("out[0][0]", out[0, :, 0, 0])
            out = torch.argmax(out, dim=1)

            print('out.shape', out.shape)
            # out_image = torch.stack([out, out, out], dim=0)
            # out_image = torch.squeeze(out_image,1)

            # out_image = np.array(out.cpu())
            y = torch.squeeze(out).numpy()
            print("y.shape", y.shape)
            plt.imshow(y)
            # plt.show()
            plt.savefig('./result/' + str(segment_name[0]) + 'result.png')
            # print('+++', np.unique(out_image))
            # out_image = label2color(out_image)
            # out_image = out_image.transpose(1, 2, 0)
            #
            # # out_image[out_image == 1] = 255
            # cv2.imwrite('./result/' + str(segment_name[0]) + 'result.png', out_image)

            # print('torch.unique(out_image)', torch.unique(out_image), out_image.shape)

    iouVal = 0
    iouVal, iou_classes = iouEvalVal.getIoU()
    print(iou_classes)
    return iouVal, iou_classes


if __name__ == "__main__":
    main()



