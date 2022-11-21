import numpy as np
from numpy.core.umath_tests import inner1d
from mkv.markovGame_badminton import MarkovGame, acts
from mkv.value_iteration import value_iteration
import pickle

def NLL(MDP, pi, demonstrated_trajectory, team, segmentation = 30):
    # print('MDP', MDP.s_a_nxs_freq)
    counter = 0
    nll_list = list()
    nll_seg = 1
    for traj in demonstrated_trajectory:
        nll_seg_list = list()
        pre_s, pre_a = traj[0]
        # print(len(traj))
        for s, a in traj[1:]:
            # if team != pre_s.split(',')[4]:
            #     pre_s, pre_a = s, a
            #     continue
            tran_prob = MDP.s_a_nxs_freq['%s+%s+%s'%(pre_s, pre_a, s)]/MDP.s_a_freq['%s+%s'%(pre_s, pre_a)]
            # pi_prob = pi[pre_s][pre_a]/sum(pi[pre_s].values())
            pi_s_sum = sum(pi[pre_s].values())
            pi_e_sum = len(acts)
            for a_tmp in pi[pre_s]:
                pi_e_sum -= 1
                pi_e_sum += np.exp(pi[pre_s][a_tmp]/pi_s_sum)
            pi_prob = pi[pre_s][pre_a]/pi_e_sum
            
            nll_seg *=  (tran_prob* pi_prob)
            pre_s, pre_a = s, a
            
            counter += 1
            if counter == segmentation:
                counter = 0
                nll_seg_list.append(-np.log(nll_seg))
                nll_seg = 1
        
        if len(nll_seg_list)>0:
            nll_list.append(sum(nll_seg_list)/len(nll_seg_list))
    # print(nll_list)
    # print(sum(nll_list)/len(nll_list))

    return sum(nll_list)/len(nll_list)

def MHD(MDP, pi, demonstrated_trajectory):
    '''
    #This function computes the Modified Hausdorff Distance (MHD) which is
    #proven to function better than the directed HD as per Dubuisson et al.
    #in the following work:
    #
    #M. P. Dubuisson and A. K. Jain. A Modified Hausdorff distance for object
    #matching. In ICPR94, pages A:566-568, Jerusalem, Israel, 1994.
    #http://ieeexplore.ieee.org/xpls/abs_all.jsp?arnumber=576361
    #
    #The function computed the forward and reverse distances and outputs the
    #maximum/minimum of both.
    #Optionally, the function can return forward and reverse distance.
    #
    #Format for calling function:
    #
    #[MHD,FHD,RHD] = ModHausdorffDist(A,B);
    #
    #where
    #MHD = Modified Hausdorff Distance.
    #FHD = Forward Hausdorff Distance: minimum distance from all points of B
    #      to a point in A, averaged for all A
    #RHD = Reverse Hausdorff Distance: minimum distance from all points of A
    #      to a point in B, averaged for all B
    #A -> Point set 1, [row as observations, and col as dimensions]
    #B -> Point set 2, [row as observations, and col as dimensions]
    #
    #No. of samples of each point set may be different but the dimension of
    #the points must be the same.
    #
    #Edward DongBo Cui Stanford University; 06/17/2014
    '''

    # def traj_generate(MDP, pi):
    #     demonstrated_trajectory

    A = np.array([s for s, a in demonstrated_trajectory])
    # B = traj_generate(MDP, pi)

    # Find pairwise distance
    D_mat = np.sqrt(inner1d(A,A)[np.newaxis].T + inner1d(B,B)-2*(np.dot(A,B.T)))
    # Calculating the forward HD: mean(min(each col))
    FHD = np.mean(np.min(D_mat,axis=1))
    # Calculating the reverse HD: mean(min(each row))
    RHD = np.mean(np.min(D_mat,axis=0))
    # Calculating mhd
    MHD = np.max(np.array([FHD, RHD]))
    return(MHD, FHD, RHD)

def cross_entropy(MDP, pi):
    ce_list = list()
    for s in MDP.s_a:
        ce = 0
        pi_s_sum = sum(pi[s].values())
        pi_e_sum = len(acts)
        for a in pi[s]:
            pi_e_sum -= 1
            pi_e_sum += np.exp(pi[s][a]/pi_s_sum)

        ### softmax
        mdp_sum = 0
        mdp_e_sum = len(acts)
        for a in MDP.s_a[s]:
            mdp_sum += MDP.s_a_freq['%s+%s'%(s, a)]
        for a in MDP.s_a[s]:
            mdp_e_sum -= 1
            mdp_e_sum += np.exp(MDP.s_a_freq['%s+%s'%(s, a)]/mdp_sum)

        s1, s2 = list(), list()
        for a in range(len(acts)):
            if str(a) in MDP.s_a[s]:
                demonstrated_prob = np.exp(MDP.s_a_freq['%s+%s'%(s, a)]/mdp_sum)/mdp_e_sum
                this_prob = np.exp(pi[s][str(a)]/pi_s_sum)/pi_e_sum
                ce += demonstrated_prob*np.log(this_prob)
            else:
                demonstrated_prob = 1/mdp_e_sum
                this_prob = np.exp(pi[s][str(a)]/pi_s_sum)/pi_e_sum if a in pi[s] else 1/pi_e_sum
                ce += demonstrated_prob*np.log(this_prob)

        # import matplotlib.pyplot as plt
        # x = np.arange(len(s1))
        # plt.plot(x, s1, 'r')
        # plt.plot(x, s2, 'b')
        # plt.show()

        ce_list.append(-ce)

    # print('ce:', sum(ce_list)/len(ce_list))
    return sum(ce_list)/len(ce_list)



