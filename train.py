# -*- coding: utf-8 -*-

import os
import torch
import torch.nn.functional as F

from utils import mkdir, get_lr, adjust_lr, Visualizer
from test import test_first_stage
from torch.nn.modules.loss import CrossEntropyLoss
import pdb
import math
import grad_cam

def structure_loss(pred, mask):
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit*wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)                          ##################################################
    inter = ((pred * mask)*weit).sum(dim=(2, 3))
    union = ((pred + mask)*weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1)/(union - inter+1)
    return (wbce + wiou).mean()


def train_first_stage(viz, writer, dataloader, net, optimizer, base_lr, criterion, criterion2, device, power, epoch, num_epochs=100):
    dt_size = len(dataloader.dataset)
    epoch_loss = 0
    step = 0
    for sample in dataloader:
        step += 1

        #print(epoch)
        #pdb.set_trace()  # 设置断点
        #print("After pause")

        img = sample[0].to(device)
        gt = sample[1].to(device)
        faz = sample[2].to(device)
        #faz = faz.unsqueeze(1)
        #print(torch.unique(gt))
        #print("gt", gt.size())
        #print("faz", faz.size())
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward
        pred,pred_faz = net(img)
        #pred = torch.sigmoid(pred)
        #pred = torch.nn.functional.normalize(pred,p=2, dim=1, eps=1e-12, out=None)
        #pred = ((pred-pred.min())/(pred.max()-pred.min()))        #pred = torch.nn.Softmax(dim = 1)(pred)
        #print("pred", pred.size())
        #print("max",pred.max(),"min",pred.min())
        #print("pred_faz", pred_faz.size())
        #print("max",pred_faz.max(),"min",pred_faz.min())
        #print(torch.unique(pred))
        viz.img(name="images", img_=img[:, :, :, :])
        viz.img(name="labels_ves", img_=gt[:, :, :, :])
        viz.img(name="prediction_ves", img_=pred[:, :, :, :])
        viz.img(name="labels_faz", img_=faz[:, :, :, :])
        viz.img(name="prediction_faz", img_=pred_faz[:, :, :, :])

        ves_loss = criterion(pred, gt)
        #ves_loss_1 = criterion(pred_1, gt)
        #ves_loss_2 = criterion(pred_2, gt)
        #ves_loss = structure_loss(pred, gt)
        #print("ves_loss", ves_loss)
        faz_loss = criterion2(pred_faz, faz)
        #print("ves_loss", ves_loss)
        #loss_ce = CrossEntropyLoss()
        #ce_loss = loss_ce(pred , gt)
        #print("ce_loss", ce_loss)
        #ce_loss_faz = loss_ce(pred_faz , faz)
        #print("ce_loss_faz", ce_loss_faz)


        #loss = ves_loss +  faz_loss
        #loss = faz_loss

        #alpha_para= 0.5+0.4*(math.sin(epoch/num_epochs*3.1415926-3.1415926/2))+0.1*((ves_loss-faz_loss)/(ves_loss + faz_loss))
        alpha_para= 0.1+0.8*(math.sin(epoch/num_epochs*3.1415926/2))+0.1*((ves_loss-faz_loss)/(ves_loss + faz_loss))
        viz.plot("alpha", alpha_para.item())


        #loss = alpha_para*ves_loss + (1-alpha_para)*faz_loss      #DWB loss   
        loss  = 0.8*ves_loss + 0.2*faz_loss
 
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



        #if (step == 1) and (epoch + 1) % 20 == 1:
            #print("start grad-cam")
            # 主任务可视化
            #grad_cam.visualize_attention(net,  writer, img,epoch, 'ves')
            # 辅助任务可视化
            #grad_cam.visualize_attention(net, writer, img,epoch, 'faz')

    print("epoch %d loss: %0.4f" % (epoch, epoch_loss))
    print("current learning rate: %f" % current_lr)

    adjust_lr(optimizer, base_lr, epoch, num_epochs, power=power)

    return net


def val_first_stage(best_metric, viz, writer, dataloader1, model, device,
                    save_epoch_freq, models_dir, results_dir, epoch, num_epochs=100):
    model.eval()
    with torch.no_grad():
        dice_arr_faz,jac_arr_faz,sen_arr_faz,spe_arr_faz,dice_arr_ves,jac_arr_ves,sen_arr_ves,spe_arr_ves = test_first_stage(viz, dataloader1, model, device, results_dir, isSave=True)

        #dice_mean = dice_arr.mean()
        dice_mean_faz = dice_arr_faz.mean()
        jac_mean_faz = jac_arr_faz.mean()
        sen_mean_faz = sen_arr_faz.mean()
        spe_mean_faz = spe_arr_faz.mean()

        dice_mean_ves = dice_arr_ves.mean()
        jac_mean_ves = jac_arr_ves.mean()
        sen_mean_ves = sen_arr_ves.mean()
        spe_mean_ves = spe_arr_ves.mean()

    # 保存模型
        mkdir(models_dir + "/exp5")
        total_metric = dice_mean_ves + dice_mean_faz
        checkpoint_path = os.path.join(models_dir, "exp5/{net}-{type}-{epoch}-{total}.pth")
        if total_metric >= best_metric["dice"]:
            best_metric["epoch"] = epoch + 1
            best_metric["dice"] = total_metric
            torch.save(model, checkpoint_path.format(net="front_model", type="regular", epoch=epoch + 1, total=total_metric))
        if (epoch + 1) % save_epoch_freq == 0:
            torch.save(model, checkpoint_path.format(net="front_model", type="regular", epoch=epoch + 1, total=total_metric))
    # if epoch == num_epochs - 1:
    #     torch.save(net, checkpoint_path.format(net="front_model", type="regular", epoch=num_epochs, total=total_metric))
    # if epoch == num_epochs - 1:
    #     torch.save(net, checkpoint_path.format(net="front_model", type="regular", epoch=num_epochs, total=total_metric))
        #viz.plot("dice rate", dice_mean)
        viz.plot("dice rate_faz", dice_mean_faz)
        viz.plot("jac rate_faz", jac_mean_faz)
        viz.plot("sen rate_faz", sen_mean_faz)
        viz.plot("spe rate_faz", spe_mean_faz)

        viz.plot("dice rate_ves", dice_mean_ves)
        viz.plot("jac rate_ves", jac_mean_ves)
        viz.plot("sen rate_ves", sen_mean_ves)
        viz.plot("spe rate_ves", spe_mean_ves)

        

        writer.add_scalars("val_dice", {"val_dice": dice_mean_ves}, epoch)
    model.train()
    # dice_arr = test_first_stage(dataloader1, model, device, results_dir, isSave=True)

    # dice_mean = dice_arr.mean()

    # # 保存模型
    # mkdir(models_dir + "/exp5")
    # total_metric = dice_mean
    # checkpoint_path = os.path.join(models_dir, "exp5/{net}-{type}-{epoch}-{total}.pth")
    # if dice_mean >= best_metric["dice"]:
    #     best_metric["epoch"] = epoch + 1
    #     best_metric["dice"] = total_metric
    #     torch.save(model, checkpoint_path.format(net="front_model", type="regular", epoch=epoch + 1, total=total_metric))
    # if (epoch + 1) % save_epoch_freq == 0:
    #     torch.save(model, checkpoint_path.format(net="front_model", type="regular", epoch=epoch + 1, total=total_metric))
    # # if epoch == num_epochs - 1:
    # #     torch.save(net, checkpoint_path.format(net="front_model", type="regular", epoch=num_epochs, total=total_metric))
    # # if epoch == num_epochs - 1:
    # #     torch.save(net, checkpoint_path.format(net="front_model", type="regular", epoch=num_epochs, total=total_metric))
    # viz.plot("dice rate", dice_mean)

    # writer.add_scalars("val_loss", {"val_loss": dice_mean}, epoch)

    return model


