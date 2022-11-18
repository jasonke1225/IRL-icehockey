import csv
from mkv.markovGame_badminton import acts

def get_events(csv_dir, f):
    events = []
    pre_s, pre_a = '', ''
    with open(csv_dir+'/'+f, newline='') as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            act          = row['type']
            # goalDiff   = row['goalDiff']
            # period     = row['period']
            player_loc   = row['player_location_area']
            opponent_loc = row['opponent_location_area']
            opponent_act = pre_a
            HorA         = row['player']
            is_point     = row['server']
            winner       = row['getpoint_player']

            # print(type(act), type(player_loc), type(opponent_loc), type(opponent_act), type(HorA))
            s = player_loc + ',' + opponent_loc + ',' + opponent_act + ',' + HorA
            a = str(acts.index(act))
            if is_point == str(1):
                s = ',,,'+HorA
            events.append((s,a))

            pre_s = s
            pre_a = a
            
            
            if is_point == str(3):
                """
                Add win state if get point
                """
                termional_state = '*,*,*,*,'+winner
                events.append((termional_state,''))
                pre_s = ''
                pre_a = ''

    return events

def curr_s_a(events, idx):
    """
    Get current state(goalDiff, manPower, period, loc, HorA) and action(act)
    """
    (s, a) = events[idx]
    return s, a

def next_s(events, idx):
    """
    For convenience, not add an end state.
    Thus the range to use this function is from 0 to idx-2
    """
    (s,a) = events[idx]
    if a == str(acts.index('goal')): # score a goal
        h_w = s[-1]
        return '*,*,*,*,'+h_w

    # next state idx+1
    (nx_s, nx_a) = events[idx+1]
    return nx_s

def extract_demonstrations(csv_dir, f, act = False, clip = True):
    """
    extract demostrations from play by play data
    goal is the end signal of an episode
    """
    events = get_events(csv_dir, f)
    trajs = []
    episode = []

    for idx in range(len(events)):
        s, a = curr_s_a(events, idx)
        episode.append((s,a)) if act else episode.append(s)
        
        if s[:-1] == '*,*,*,*,':
            trajs.append(episode)
            episode = []
        else:
            continue
    if episode != []:
        trajs.append(episode)

    # trajs_select = [episode for episode in trajs if len(episode)>150]
    trajs_select = [episode for episode in trajs]

    # it is optinal if you want to make trajs shorter
    if clip:
        trajs_select = [episode[-150:] for episode in trajs_select]

    return trajs_select

def test_extract_demonstrations():
    d = {}
    csv_dir = '/home/jasonke/桌面/data_science_project/IRL-icehockey/Slgq/data'
    file_all = os.listdir(csv_dir)
    for file in file_all:
        trajs = extract_demonstrations(csv_dir, file)
        episode_1 = [episode[0] for episode in  trajs]
        for e in episode_1:
            goal_diff, _,_,_,_ = e.split(',')
            if goal_diff in d:
                d[goal_diff] += 1
            else:
                d[goal_diff] = 1
    print(d)

if __name__ == '__main__':
    import os
    test_extract_demonstrations()