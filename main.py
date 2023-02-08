import argparse
import os

import pandas as pd
import torch
import torch.optim as optim
from thop import profile, clever_format
from torch.utils.data import DataLoader
from tqdm import tqdm

import data
from model import AWCL_model

import json
from losses import SupConLoss
os.environ['LOCAL_DATA_DIR']='/data/path/to/the/training/set/PULSE/'
os.environ['CUDA_VISIBLE_DEVICES']='0'

# train for one epoch with the proposed AWCL loss
def train(net, data_loader, train_optimizer, temperature):
    net.train()
    CritAWCL=SupConLoss(temperature).cuda()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    #start training
    for pos_1, pos_2, target in train_bar:
        pos_1, pos_2, target = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True), target.cuda(non_blocking=True)
        _, out_1 = net(pos_1)
        _, out_2 = net(pos_2)
        # construct features
        features = torch.stack([out_1, out_2], dim=1)

        # compute loss
        if target[0]==-1:#anatomy information not available
            #use the regular contra loss
            loss = CritAWCL(features)#regular contrastive
        else:
            #use the anatomy-aware contra loss
            loss = CritAWCL(features, target)#AWCL

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train AWCL')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--batch_size', default=32, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs over the dataset to train')

    # args parse
    args = parser.parse_args()
    feature_dim, temperature = args.feature_dim, args.temperature
    batch_size, epochs = args.batch_size, args.epochs

    # data prepare
    with open('pretrain_data_list.json','r') as jsonfile:#list of the training data file
        jlist= json.load(jsonfile)
    #data loader
    train_data = data.PULSELoader(scan_id_list=jlist, config_file='spdata_SPD.yml', transform=data.train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True, drop_last=True)

    # model setup and optimizer config
    model = AWCL_model(base_model='resnet18', out_dim=feature_dim).cuda()

    flops, params = profile(model, inputs=(torch.randn(1, 3, 227, 227).cuda(),))
    flops, params = clever_format([flops, params])
    print('# Model Params: {} FLOPs: {}'.format(params, flops))
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    # training loop
    results = {'train_loss': []}

    save_name_pre = 'AWCL_feat{}_temp{}_bs{}_ep{}'.format(feature_dim, temperature, batch_size, epochs)
    if not os.path.exists('results'):
        os.mkdir('results')

    start_epoch=1
    for epoch in range(start_epoch, epochs + 1):
        train_loss = train(model, train_loader, optimizer, temperature)
        results['train_loss'].append(train_loss)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(start_epoch, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')
        
        torch.save(model.state_dict(), 'results/{}_model_ep{}.pth'.format(save_name_pre, epoch))
