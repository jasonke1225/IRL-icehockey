import os
import csv

acts = ['等待發球', '切球', '勾球', '小平球', '平球', '後場抽平球', '挑球', '推球', '撲球', '擋小球', '放小球',
       '未知球種', '殺球', '發短球', '發長球', '過度切球', '長球', '防守回抽', '防守回挑', '點扣']

class MarkovGame(object):

    def __init__(self, csv_dir, ignore_team='B'):
        self.csv_dir = csv_dir
        self.acts = acts
        self.team = ignore_team
        self.trans = self._build_transition(csv_dir)
        self._decomposition()

    def _build_transition(self, csv_dir):
        print('###### Building Markov Game Model from data ######')
        """
        trans is a dict(), key is the current state, value is a dict(),
              where key is action + next_state, value is the occurrence number in our data
        """
        trans = {}

        def insert2dict(s, a, nx_s):
            # print(s, a, nx_s)
            # input()
            if s in trans:
                to_dict = trans[s]
                key = a + '+' + nx_s
                if key in to_dict:
                    to_dict[key] = to_dict[key] + 1
                else:
                    to_dict[key] = 1
            else:
                to_dict = {}
                key = a + '+' + nx_s
                to_dict[key] = 1
                trans[s] = to_dict

        file_all = os.listdir(csv_dir)
        file_all.sort()
        for f in file_all:
            # first check if data in gameTime increasing order
            # check_csv_seq(csv_dir, f) 
            print('check data for correct order')
            pre_s, pre_a, oppo_act = '', '', ''
            with open(csv_dir+'/'+f, newline='') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    act          = row['type']
                    # goalDiff   = row['goalDiff']
                    # period     = row['period']
                    player_loc   = row['player_location_area']
                    opponent_loc = row['opponent_location_area']
                    opponent_act = oppo_act
                    HorA         = row['player']
                    is_point     = row['server']
                    winner       = row['getpoint_player']

                    if pre_s == '' and pre_a == '' and winner is not '':
                        pre_s = ',,,'+HorA
                        pre_a = '0'
                        insert2dict(pre_s, pre_a, '*,*,*,' + winner)
                        pre_s = ''
                        pre_a = ''
                        continue

                    if is_point == str(1) and HorA == self.team:
                        oppo_act = str(self.acts.index(act))
                        ### 對手發球
                        pre_s = ',,,'+HorA
                        pre_a = '0'
                        continue

                    elif HorA == self.team:
                        oppo_act = str(self.acts.index(act))
                        if is_point == str(3):
                            insert2dict(pre_s, pre_a, '*,*,*,' + winner)
                            pre_s = ''
                            pre_a = ''
                        continue

                    # print(type(act), type(player_loc), type(opponent_loc), type(opponent_act), type(HorA))
                    s = player_loc + ',' + opponent_loc + ',' + opponent_act + ',' + HorA
                    a = str(self.acts.index(act))
                    
                    if pre_s == '' and pre_a == '':
                        pass
                    else:
                        insert2dict(pre_s, pre_a, s)
                    pre_s = s
                    pre_a = a
                    
                    if is_point == str(1):
                        pre_s = ',,,'+HorA
                    elif is_point == str(3):
                        """
                        Add win state if get point
                        """
                        insert2dict(pre_s, pre_a, '*,*,*,' + winner)
                        pre_s = ''
                        pre_a = ''
                    
        return trans

    def _decomposition(self):
        print('######              decomposing             ######')
        pre_s        = {} # a dict: {state : [list of previous state]}
        s_a          = {} # a dict: {state : [list of actions]}
        s_a_freq     = {} # a ditt: {state,action : frequency}
        s_a_nxs      = {} # a dict: {state,action : [list of next state]}
        s_a_nxs_freq = {} # a dict: {state,action,nx state : frequency}
        
        for s in self.trans.keys():
            pre_s[s] = []
        for s in self.trans.keys():
            to_dict = self.trans[s]
            to_keys = to_dict.keys()
            for to_key in to_keys:
                a, nxs = to_key.split('+')
                num    = to_dict[to_key]

                # update pre_s
                if nxs in pre_s:
                    if s not in pre_s[nxs]:
                        pre_s[nxs].append(s)
                else:
                    pre_s[nxs] = [s]

                # update s_a
                if s in s_a:
                    action_list = s_a[s]
                    if a not in action_list:
                        s_a[s].append(a)
                else:
                    s_a[s] = [a]

                # update s_a_freq
                s_and_a = s + '+' + a
                if s_and_a in s_a_freq:
                    s_a_freq[s_and_a] += num
                else:
                    s_a_freq[s_and_a]  = num

                # update s_a_nxs
                s_and_a = s + '+' + a
                if s_and_a in s_a_nxs:
                    next_state_list = s_a_nxs[s_and_a]
                    if nxs not in next_state_list:
                        s_a_nxs[s_and_a].append(nxs)
                else:
                    s_a_nxs[s_and_a] = [nxs]

                # update s_a_nxs_freq
                s_and_a_and_nxs = s + '+' + a + '+' + nxs
                if s_and_a_and_nxs in s_a_nxs_freq:
                    s_a_nxs_freq[s_and_a_and_nxs] += num
                else:
                    s_a_nxs_freq[s_and_a_and_nxs]  = num

        tmp = [s for s in self.trans.keys()]
        tmp.append('*,*,*,A')
        tmp.append('*,*,*,B')

        self.s            = tmp
        self.end_s        = ['*,*,*,A','*,*,*,B']
        self.s2idx        = {tmp[i]:i for i in range(len(tmp))}
        self.pre_s        = pre_s
        self.s_a          = s_a
        self.s_a_freq     = s_a_freq
        self.s_a_nxs      = s_a_nxs
        self.s_a_nxs_freq = s_a_nxs_freq  

    def _get_nxs_and_prob(self, s, a):
        """
        get all the (next state, probablity) pair
        if taking action (a) at state (s)
        """
        k1 = '%s+%s'%(s, a)
        freq     = self.s_a_freq[k1]
        nxs_list = self.s_a_nxs[k1]
        nxs_and_prob = []

        for nxs in nxs_list:
            k2 = '%s+%s+%s'%(s, a, nxs)
            this_freq = self.s_a_nxs_freq[k2]
            this_prob = float(this_freq) / float(freq)
            nxs_and_prob.append([nxs, this_prob])
        
        return nxs_and_prob

    def get_trans_prob(self, s, a, nx_s):
        """
        The transition probability of landing at next state
        when taking action (a) at state (s)
        """
        if s == '*,*,*,*,H' or s == '*,*,*,*,A':
            return 0
        if a not in self.s_a[s]:
            return 0

        nxs_and_prob = self._get_nxs_and_prob(s, a)
        for nxs, prob in nxs_and_prob:
            if nxs == nx_s:
                return prob

        return 0

    def get_act(self, s):
        return self.s_a[s]
    
    def get_nxs(self, s, a):
        key = '%s+%s'%(s, a)
        return self.s_a_nxs[key]
                
