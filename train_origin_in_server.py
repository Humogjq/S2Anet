# -*- coding: utf-8 -*-

import os
import torch

from utils import mkdir, get_lr, adjust_lr, Visualizer
from test import test_first_stage
from torch.nn.modules.loss import CrossEntropyLoss


def train_first_stage(viz, writer, dataloader, net, optimizer, base_lr, criterion, criterion2, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1
        img = sample[0].to(device)
        gt = sample[1].to(device)
        #print(torch.unique(gt))
        print("gt", gt.size())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        pred = net(img)
        #pred = torch.sigmoid(pred)
        #pred = torch.nn.functional.normalize(pred,p=2, dim=1, eps=1e-12, out=None)
        #pred = ((pred-pred.min())/(pred.max()-pred.min()))        #pred = torch.nn.Softmax(dim = 1)(pred)
        print("pred", pred.size())
        print("max",pred.max(),"min",pred.min())
        #print(torch.unique(pred))
        viz.img(name="images", img_=img[:, :, :, :])
        viz.img(name="labels", img_=gt[:, :, :, :])
        viz.img(name="prediction", img_=pred[:, :, :, :])
        dice_loss = criterion(pred, gt)

        smooth_loss = criterion2(pred, gt)
        loss_ce = CrossEntropyLoss()
        ce_loss = loss_ce(pred , gt)


        loss = dice_loss 

        #loss = dice_loss
        loss.backward(retain_graph=True)
        optimizer.step()
        epoch_loss += loss.item()

        # 当前batch图像的loss
        niter = epoch * len(dataloader) + step
        writer.add_scalars("train_loss", {"train_loss": loss.item()}, niter)
        print("%d / %d, train loss: %0.4f" % (step, (dt_size - 1) // dataloader.batch_size + 1, loss.item()))
        viz.plot("train loss", loss.item())

        # 写入当前lr
        current_lr = get_lr(optimizer)
        viz.plot("learning rate", current_lr)
        writer.add_scalars("learning_rate", {"lr": current_lr}, niter)

    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)

    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)

    return net


def val_first_stage(best_metric, viz, writer, dataloader1, model, device,
                    save_epoch_freq, models_dir, results_dir, epoch, num_epochs=100):
    model.eval()
    dice_arr = test_first_stage(dataloader1, model, device, results_dir, isSave=True)

    dice_mean = dice_arr.mean()

    # 保存模型
    mkdir(models_dir + "/exp5")
    total_metric = dice_mean
    checkpoint_path = os.path.join(models_dir, "exp5/{net}-{type}-{epoch}-{total}.pth")
    if dice_mean >= best_metric["dice"]:
        best_metric["epoch"] = epoch + 1
        best_metric["dice"] = total_metric
        torch.save(model, checkpoint_path.format(net="front_model", type="regular", epoch=epoch + 1, total=total_metric))
    if (epoch + 1) % save_epoch_freq == 0:
        torch.save(model, checkpoint_path.format(net="front_model", type="regular", epoch=epoch + 1, total=total_metric))
    # if epoch == num_epochs - 1:
    #     torch.save(net, checkpoint_path.format(net="front_model", type="regular", epoch=num_epochs, total=total_metric))
    # if epoch == num_epochs - 1:
    #     torch.save(net, checkpoint_path.format(net="front_model", type="regular", epoch=num_epochs, total=total_metric))
    viz.plot("dice rate", dice_mean)

    writer.add_scalars("val_loss", {"val_loss": dice_mean}, epoch)

    return model


