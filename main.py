import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from generator import Generator
from classify import *
from utils import * 
from SAC import Agent
from attack import inversion

parser = argparse.ArgumentParser(description="RLB-MI")
parser.add_argument('-model_name', default='VGG16')
parser.add_argument("-max_episodes", type=int, default=40000)
parser.add_argument("-max_step", type=int, default=1)
parser.add_argument("-seed", type=int, default=42)
parser.add_argument("-alpha", type=float, default=0)
parser.add_argument("-n_classes", type=int, default=1000)
parser.add_argument("-z_dim", type=int, default=100)
parser.add_argument("-n_target", type=int, default=100)
args = parser.parse_args()

if __name__ == "__main__":
    model_name = args.model_name
    max_episodes = args.max_episodes
    max_step = args.max_step
    seed = args.seed
    alpha = args.alpha
    n_classes = args.n_classes
    z_dim = args.z_dim
    n_target = args.n_target

    print("Target Model : " + model_name)
    G = Generator(z_dim)
    G = nn.DataParallel(G).cuda()
    G = G.cuda()
    ckp_G = torch.load('weights/CelebA.tar')['state_dict']
    load_my_state_dict(G, ckp_G)
    G.eval()

    if model_name == "VGG16":
        T = VGG16(n_classes)
        path_T = './weights/VGG16.tar'
    elif model_name == 'ResNet-152':
        T = IR152(n_classes)
        path_T = './weights/ResNet-152.tar'
    elif model_name == "Face.evoLVe":
        T = FaceNet64(n_classes)
        path_T = './weights/Face.evoLVe.tar'

    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(path_T)
    T.load_state_dict(ckp_T['state_dict'], strict=False)
    T.eval()

    E = FaceNet(n_classes)
    path_E = './weights/Eval.tar'
    E = torch.nn.DataParallel(E).cuda()
    ckp_E = torch.load(path_E)
    E.load_state_dict(ckp_E['state_dict'], strict=False)
    E.eval()

    def seed_everything(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)  # type: ignore
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = True  # type: ignore

    seed_everything(seed)

    total = 0
    cnt = 0
    cnt5 = 0

    identities = range(n_classes)
    targets = random.sample(identities, n_target)

    for i in targets:
        agent = Agent(state_size=z_dim, action_size=z_dim, random_seed=seed, hidden_size=256, action_prior="uniform")
        recon_image = inversion(agent, G, T, alpha, z_dim=z_dim, max_episodes=max_episodes, max_step=max_step, label=i, model_name=model_name)
        _, output= E(low2high(recon_image))
        eval_prob = F.softmax(output[0], dim=-1)
        top_idx = torch.argmax(eval_prob)
        _, top5_idx = torch.topk(eval_prob, 5)

        total += 1
        if top_idx == i:
            cnt += 1
        if i in top5_idx:
            cnt5 += 1

        acc = cnt / total
        acc5 = cnt5 / total
        print("Classes {}/{}, Accuracy : {:.3f}, Top-5 Accuracy : {:.3f}".format(total, n_target, acc, acc5))
    
