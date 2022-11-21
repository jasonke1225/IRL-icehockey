import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
# from mkv.markovGame import MarkovGame
from mkv.markovGame_badminton import MarkovGame
from mkv.value_iteration import value_iteration
from irl.maxent_irl import maxent_irl
# from utils.extract import extract_demonstrations
from utils.extract_badminton import extract_demonstrations
from utils.metric import NLL, cross_entropy

def draw(y=list(), title=str(), file_len = 1):
    x = np.arange(len(y)/file_len)
    if file_len > 1:
        for idx in range(file_len):
            plt.figure(idx)
            plt.plot(x, y[idx::file_len])
            plt.title(title) 
        plt.show()
    else:
        plt.plot(x,y)
        plt.title(title) 
        plt.show()

def run(csv_dir, mg, HorA, save_dir, deter, ignore_team):
    # build feature matrix
    N_STATES = len(mg.s)
    print(N_STATES)
    feat_map = np.eye(N_STATES)
    # domain konwledge reward
    rbg = [1. for s in mg.s]
    if ignore_team == 'B':
        rbg[mg.s2idx[mg.end_s[0]]] = 2.
        rbg[mg.s2idx[mg.end_s[1]]] = 0.
    if ignore_team == 'A':
        rbg[mg.s2idx[mg.end_s[0]]] = 0.
        rbg[mg.s2idx[mg.end_s[1]]] = 2.

    theta = rbg.copy()
    gamma = 0.9
    lr = 0.001
    n_iters = 100
    if not os.path.exists(save_dir+'/'+HorA):
        os.mkdir(save_dir+'/'+HorA)
    
    ce_list = list()
    nll_list = list()
    grad_list = list()
    # train
    file_all = os.listdir(csv_dir)
    file_all.sort()
    for iter in range(n_iters):
        traj = list()
        grad = 0
        for i in range(len(file_all)):
            print("#### Game ", str(i+1), " out of ", str(len(file_all)), " | iter ", str(iter+1), " ####")
            trajs = extract_demonstrations(csv_dir, file_all[i])
            if trajs == []:
                continue
            theta, reward, gradient = maxent_irl(feat_map, mg, gamma, trajs, theta, rbg, lr, deter)
            grad += np.sqrt(gradient*gradient).mean()
            
            # save
            save_theta = save_dir + '/' + HorA + '/' + 'iter_' + str(iter) + '_aft_game_' + str(i) + '_theta'
            save_reward = save_dir + '/' + HorA + '/'+ 'iter_' + str(iter) + '_aft_game_' + str(i) + '_reward'
            with open(save_theta, 'wb') as f:
                pickle.dump(theta, f)
            with open(save_reward, 'wb') as f:
                pickle.dump(reward, f)
            
            traj_tmp = extract_demonstrations(csv_dir, file_all[i], act=True, clip=True, ignore_team=ignore_team)
            traj = traj+traj_tmp 
        
        reward = np.dot(feat_map, theta)
        state_value, policy = value_iteration(mg, reward, gamma, error=0.01, deterministic=False)

        ce = cross_entropy(mg, policy)
        nll = NLL(mg, policy, traj, team=HorA, segmentation = 30)
        
        ce_list.append(ce)
        nll_list.append(nll)
        grad_list.append(grad/len(file_all))
        # grad_list.append(np.sqrt(gradient*gradient).mean())
        # print(grad_list)
        
    draw(nll_list, 'NLL Metric per iteration')
    draw(ce_list, 'Cross Entropy Metric per iteration')
    draw(grad_list, 'Gradient per iteration')
        
if __name__ == '__main__':
    # csv_dir = '/home/jasonke/桌面/data_science_project/IRL-icehockey/Slgq/data'
    # save_dir = '/home/jasonke/桌面/data_science_project/IRL-icehockey/Slgq/save_reward'
    csv_dir = 'data/chou_tien_chen/'
    save_dir = 'data/chou_tranfers_result/'
    mg = MarkovGame(csv_dir, 'B')
    d = mg.__dict__
    ks = mg.__dict__.keys()
    # print(ks)
    # for k in ks:
    #     print(k)
    #     try:
    #         print(d[k].keys()[0], d[k].values()[0])
    #     except:
    #         print(d[k])
    run(csv_dir, mg, 'Home', save_dir, True, 'B')
    # run(csv_dir, mg, 'Away', save_dir, True)