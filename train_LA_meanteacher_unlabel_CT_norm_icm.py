# 纯的MT
import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.CTdataset import CTdataset, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler
import torch.nn as nn
from torch.nn.parameter import Parameter
from test_util_CT import test_all_case
import os
import pandas as pd
# 定义设备位置
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "6"
# device = torch.device('cuda:0')  # 定义gpu位置
torch.set_num_threads(4)

class Linear_vector(nn.Module):
    def __init__(self, n_dim):
        super(Linear_vector, self).__init__()
        self.n_dim = n_dim
        self.paras = Parameter(torch.Tensor(self.n_dim, self.n_dim))   # linear coefficients
        self.init_ratio = 1e-3   
        self.initialize()   
    
    def initialize(self):
        for param in self.paras:
            param.data.normal_(0, self.init_ratio)
    
    def forward(self, x):
        # result = paras*x   (16*9)=(16*16)*(16*9)
        result = torch.mm(self.paras, x)
        return result

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='/data1/mjz/semi/Pancreas-CT/data_new_norm/', help='Name of Experiment')
# parser.add_argument('--root_path', type=str, default='/data/mjz/semi/dataset/Pancreas-CT/data_new_norm/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='MT_unlabel_CT_norm_noaug_icm_20230529', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
### costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,  default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')

# steps
parser.add_argument('--max_step', type=float,
                    default=60, help='consistency_rampup') 
parser.add_argument('--min_step', type=float,
                    default=60, help='consistency_rampup')
parser.add_argument('--start_step1', type=float,
                    default=100, help='consistency_rampup')
parser.add_argument('--start_step2', type=float,
                    default=100, help='consistency_rampup')
parser.add_argument('--coefficient', type=float,
                    default=1.0, help='consistency_rampup')

args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "/data1/mjz/semi/UA-MT-master/model/" + args.exp + "/"
# snapshot_path = "/data/mjz/semi/CauSSL_code/model/" + args.exp + "/"


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (96, 96, 96)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # if os.path.exists(snapshot_path + '/code'):
    #     shutil.rmtree(snapshot_path + '/code')
    # shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git','__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    db_train = CTdataset(base_dir=train_data_path,
                       split='train',
                       transform = transforms.Compose([
                        #   RandomRotFlip(),
                          RandomCrop(patch_size),
                          ToTensor(),
                          ]))
    db_test = CTdataset(base_dir=train_data_path,
                       split='test',
                       transform = transforms.Compose([
                           CenterCrop(patch_size),
                           ToTensor()
                       ]))
    labeled_idxs = list(range(12))
    unlabeled_idxs = list(range(12, 62))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size-labeled_bs)
    def worker_init_fn(worker_id):
        random.seed(args.seed+worker_id)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=2, pin_memory=True, worker_init_fn=worker_init_fn)

    model.train()
    ema_model.train()
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    linear_paras1 = [] 
    linear_paras2 = []  
    count = 0
    for name, parameters in model.named_parameters():
        print(name)
        if 'conv' in name and 'weight' in name :
            if len(parameters.shape) == 5:
                count += 1
                outdim = parameters.shape[0] # output dimension
                linear_paras1.append(Linear_vector(outdim))
                linear_paras2.append(Linear_vector(outdim))

    linear_paras1 = nn.ModuleList(linear_paras1)
    linear_paras2 = nn.ModuleList(linear_paras2)
    linear_paras1 = linear_paras1.cuda()
    linear_paras2 = linear_paras2.cuda()
    linear_optimizer1 = torch.optim.Adam(linear_paras1.parameters(), 2e-2)
    linear_optimizer2 = torch.optim.Adam(linear_paras2.parameters(), 2e-2)  

    if args.consistency_type == 'mse':
        consistency_criterion = losses.softmax_mse_loss
    elif args.consistency_type == 'kl':
        consistency_criterion = losses.softmax_kl_loss
    else:
        assert False, args.consistency_type

    writer = SummaryWriter(snapshot_path+'/log')
    logging.info("{} itertations per epoch".format(len(trainloader)))

    iter_num = 0
    iter_num_max = 0
    max_epoch = max_iterations//len(trainloader)+1
    lr_ = base_lr
    model.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, sampled_batch in enumerate(trainloader):
            time2 = time.time()
            if iter_num > args.start_step1 and iter_num%args.min_step == 0:
                for i_max in range(args.max_step):
                    # optimize linear coefficients
                    icm_loss1 = -losses.l_correlation_cos_mean(model, ema_model, linear_paras1)
                    icm_loss2 = -losses.l_correlation_cos_mean(ema_model, model, linear_paras2)

                    linear_optimizer1.zero_grad()
                    linear_optimizer2.zero_grad()

                    icm_loss1.backward()
                    icm_loss2.backward()

                    linear_optimizer1.step()
                    linear_optimizer2.step()

                    iter_num_max += 1

                    writer.add_scalar('loss/icm_loss1_max', -icm_loss1, iter_num_max)
                    writer.add_scalar('loss/icm_loss2_max', -icm_loss2, iter_num_max)
            # print('fetch data cost {}'.format(time2-time1))
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            ema_inputs = unlabeled_volume_batch + noise
            outputs = model(volume_batch)
            with torch.no_grad():
                ema_output = ema_model(ema_inputs)
            # T = 8
            # volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            # stride = volume_batch_r.shape[0] // 2
            # preds = torch.zeros([stride * T, 2, 112, 112, 80]).cuda()
            # for i in range(T//2):
            #     ema_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
            #     with torch.no_grad():
            #         preds[2 * stride * i:2 * stride * (i + 1)] = ema_model(ema_inputs)
            # preds = F.softmax(preds, dim=1)
            # preds = preds.reshape(T, stride, 2, 112, 112, 80)
            # preds = torch.mean(preds, dim=0)  #(batch, 2, 112,112,80)
            # uncertainty = -1.0*torch.sum(preds*torch.log(preds + 1e-6), dim=1, keepdim=True) #(batch, 1, 112,112,80)


            ## calculate the loss
            loss_seg = F.cross_entropy(outputs[:labeled_bs], label_batch[:labeled_bs])
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = losses.dice_loss(outputs_soft[:labeled_bs, 1, :, :, :], label_batch[:labeled_bs] == 1)
            supervised_loss = 0.5*(loss_seg+loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num//150)
            consistency_dist = consistency_criterion(outputs[labeled_bs:], ema_output) #(batch, 2, 112,112,80)
            # threshold = (0.75+0.25*ramps.sigmoid_rampup(iter_num, max_iterations))*np.log(2)
            # mask = (uncertainty<threshold).float()
            # consistency_dist = torch.sum(mask*consistency_dist)/(2*torch.sum(mask)+1e-16)
            consistency_dist = torch.mean(consistency_dist)
            consistency_loss = consistency_weight * consistency_dist

            if iter_num > args.start_step2 and iter_num_max > 0:
                icm_loss1 = losses.l_correlation_cos_mean(model, ema_model, linear_paras1)
                # icm_loss2 = losses.l_correlation_cos_mean(model2, model1, linear_paras2)
            else:
                icm_loss1 = 0
                # icm_loss2 = 0

            loss = supervised_loss + consistency_loss + args.coefficient*icm_loss1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)

            iter_num = iter_num + 1
            # writer.add_scalar('uncertainty/mean', uncertainty[0,0].mean(), iter_num)
            # writer.add_scalar('uncertainty/max', uncertainty[0,0].max(), iter_num)
            # writer.add_scalar('uncertainty/min', uncertainty[0,0].min(), iter_num)
            # writer.add_scalar('uncertainty/mask_per', torch.sum(mask)/mask.numel(), iter_num)
            # writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)
            writer.add_scalar('loss/icm_loss1_min', icm_loss1, iter_num)

            logging.info('iteration %d : loss : %f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), consistency_dist.item(), consistency_weight))
            # if iter_num % 50 == 0:
            #     image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/Image', grid_image, iter_num)

            #     # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
            #     image = utils.decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('train/Predicted_label', grid_image, iter_num)

            #     image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
            #     writer.add_image('train/Groundtruth_label', grid_image, iter_num)

            #     image = uncertainty[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/uncertainty', grid_image, iter_num)

            #     mask2 = (uncertainty > threshold).float()
            #     image = mask2[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('train/mask', grid_image, iter_num)
            #     #####
            #     image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     grid_image = make_grid(image, 5, normalize=True)
            #     writer.add_image('unlabel/Image', grid_image, iter_num)

            #     # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
            #     image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
            #     image = utils.decode_seg_map_sequence(image)
            #     grid_image = make_grid(image, 5, normalize=False)
            #     writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

            #     image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
            #     grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
            #     writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)

            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

                # save_mode_path = os.path.join(snapshot_path, 'iter_saveT_' + str(iter_num) + '.pth')
                # torch.save(ema_model.state_dict(), save_mode_path)
                # logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_'+str(max_iterations)+'.pth')
    torch.save(model.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()

# testing
    with open('/Pancreas-CT/test.list', 'r') as f:
        image_list = f.readlines()
    image_list = [args.root_path +item.replace('\n', '')+".h5" for item in image_list]


    def test_calculate_metric_1(epoch_num):
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
        save_mode_path = os.path.join(snapshot_path, 'iter_' + str(epoch_num) + '.pth')
        net.load_state_dict(torch.load(save_mode_path))
        print("init weight from {}".format(save_mode_path))
        net.eval()

        avg_metric = test_all_case(net, image_list, num_classes=num_classes,
                                patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
                                save_result=False)
        return avg_metric
    
    # def test_calculate_metric_2(epoch_num):
    #     net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=False).cuda()
    #     save_mode_path = os.path.join(snapshot_path, 'iter2_' + str(epoch_num) + '.pth')
    #     net.load_state_dict(torch.load(save_mode_path))
    #     print("init weight from {}".format(save_mode_path))
    #     net.eval()

    #     avg_metric = test_all_case(net, image_list, num_classes=num_classes,
    #                             patch_size=(96, 96, 96), stride_xy=16, stride_z=16,
    #                             save_result=False)

    #     return avg_metric

    nums = [1000, 2000, 3000, 4000, 5000, 6000]
    first_list = np.zeros([6, 4])
    second_list = np.zeros([6, 4])
    count = 0
    for i in nums:
        metric1 = test_calculate_metric_1(i)
        first_list[count, :] = metric1
        # metric2 = test_calculate_metric_2(i)

        # second_list[count, :] = metric2
        count += 1

    write_csv = "/test_csv/" + args.exp + "_max_" + str(args.max_step) + "_min_" + str(args.min_step) + "_start1_" + str(args.start_step1) + "_start2_" + str(args.start_step2) + "_coe_" + str(args.coefficient) +  ".csv"
    save = pd.DataFrame({'dice':first_list[:,0], 'jc':first_list[:,1], 'hd95':first_list[:,2], 'asd':first_list[:,3]})
    save.to_csv(write_csv, index=False, sep=',')

    # write_csv = "/test_csv/" + args.exp + "_max_" + str(args.max_step) + "_min_" + str(args.min_step) + "_start1_" + str(args.start_step1) + "_start2_" + str(args.start_step2) + "_coe_" + str(args.coefficient) + "_2.csv"
    # save = pd.DataFrame({'dice':second_list[:,0], 'jc':second_list[:,1], 'hd95':second_list[:,2], 'asd':second_list[:,3]})
    # save.to_csv(write_csv, index=False, sep=',')