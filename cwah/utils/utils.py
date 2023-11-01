import json
import argparse
from datetime import datetime
import os
import numpy as np
import pdb
import re
import math
import torch
from torch.utils.tensorboard import SummaryWriter


def parse_prog(prog):
    program = []
    actions = []
    o1 = []
    o2 = []
    for progstring in prog:
        params = []

        patt_action = r'^\[(\w+)\]'
        patt_params = r'\<(.+?)\>\s*\((.+?)\)'

        action_match = re.search(patt_action, progstring.strip())
        action_string = action_match.group(1).upper()

        param_match = re.search(patt_params, action_match.string[action_match.end(1):])
        while param_match:
            params.append((param_match.group(1), int(param_match.group(2))))
            param_match = re.search(patt_params, param_match.string[param_match.end(2):])

        program.append((action_string, params))
        actions.append(action_string)
        if len(params) > 0:
            o1.append(params[0])
        else:
            o1.append(None)
        if len(params) > 1:
            o2.append(params[1])
        else:
            o2.append(None)

    return actions, o1, o2


def LCS(X, Y):
    # find the length of the strings
    m = len(X)
    n = len(Y)

    # declaring the array for storing the dp values
    L = [[None] * (n + 1) for i in range(m + 1)]

    """Following steps build L[m + 1][n + 1] in bottom up fashion 
    Note: L[i][j] contains length of LCS of X[0..i-1] 
    and Y[0..j-1]"""
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                L[i][j] = 0
            elif X[i - 1] == Y[j - 1]:
                L[i][j] = L[i - 1][j - 1] + 1
            else:
                L[i][j] = max(L[i - 1][j], L[i][j - 1])

                # L[m][n] contains the length of LCS of X[0..n-1] & Y[0..m-1]
    return L[m][n]*1./(max(m,n)+1.0e-5)


# end of function lcs
def computeLCS_multiple(gt_programs, pred_programs):
    lcs_action = []
    lcs_o1 = []
    lcs_o2 = []
    lcs_instr = []
    for it in range(len(gt_programs)):

        lcsa, o1, o2, instr = computeLCS(gt_programs[it], pred_programs[it])
        lcs_action.append(lcsa)
        lcs_o1.append(o1)
        lcs_o2.append(o2)
        lcs_instr.append(instr)
    return np.mean(lcs_action), np.mean(lcs_o1), np.mean(lcs_o2), np.mean(lcs_instr)

def computeLCS(gt_prog, pred_prog):
    if 'stop' in gt_prog[0]:
        stop_index_gt = [it for it, x in enumerate(gt_prog[0]) if x == 'stop'][0]
        gt_prog = [x[:stop_index_gt] for x in gt_prog]

    if 'stop' in pred_prog[0]:
        stop_index_pred = [it for it, x in enumerate(pred_prog[0]) if x == 'stop'][0]
        pred_prog = [x[:stop_index_pred] for x in pred_prog]


    gt_program = list(zip(gt_prog[0], gt_prog[1], gt_prog[2]))
    pred_program = list(zip(pred_prog[0], pred_prog[1], pred_prog[2]))
    action = LCS(gt_prog[0], pred_prog[0])
    obj1 = LCS(gt_prog[1], pred_prog[1])
    obj2 = LCS(gt_prog[2], pred_prog[2])
    instr = LCS(gt_program, pred_program)

    return action, obj1, obj2, instr

class DictObjId:
    def __init__(self, elements=None, include_other=True):
        self.el2id = {}
        self.id2el = []
        self.include_other = include_other
        if include_other:
            self.el2id = {'other': 0}
            self.id2el = ['other']
        if elements:
            for element in elements:
                self.add(element)

    def get_el(self, id):
        if self.include_other and id >= len(self.id2el):
            return self.id2el[0]
        else:
            return self.id2el[id]

    def get_id(self, el):
        el = el.lower()
        if el in self.el2id.keys():
            return self.el2id[el]
        else:
            if self.include_other:
                return 0
            else:
                return self.el2id[el]

    def add(self, el):
        el = el.lower()
        if el not in self.el2id.keys():
            num_elems = len(self.id2el)
            self.el2id[el] = num_elems
            self.id2el.append(el)

    def __len__(self):
        return len(self.id2el)


class Helper:
    def __init__(self, args, dir_name=None):
        self.args = args
        self.dir_name = dir_name
        self.setup()

    def setup(self):
        if self.args.interactive:
            return
        argvars = vars(self.args)
        names_save = ['dataset_folder', 'pomdp', 'graphsteps', 'training_mode', 'invertedge']
        names_and_params = [(x, argvars[x]) for x in names_save]
        if self.args.debug:
            fname = 'debug'

        else:
            if self.dir_name is None:
                param_name = '_'.join(['{}.{}'.format(x, y) for x, y in names_and_params])

                if argvars['training_mode'] != 'bc':
                    param_name += '/offp.{}_eps.{}_gamma.{}'.format(self.args.off_policy, self.args.eps_greedy, self.args.gamma)

                fname = str(datetime.now()).replace(' ', '_').replace(':', '.')
                self.dir_name = '{}/{}/{}'.format(self.args.log_dir, param_name, fname)
                os.makedirs(self.dir_name, exist_ok=True)
                with open('{}/args.txt'.format(self.dir_name), 'w+') as f:
                    args_str = str(self.args)
                    f.writelines(args_str)

            self.log_dir_name = '{}/{}'.format(self.dir_name, 'log')
            self.writer = SummaryWriter(self.log_dir_name)

    def save(self, epoch, loss_avg, model_params, optim_params):
        dir_chkpt = '{}/chkpt'.format(self.dir_name)
        if not os.path.isdir(dir_chkpt):
            os.makedirs(dir_chkpt, exist_ok=True)
        torch.save({
            'epoch': epoch,
            'loss': loss_avg,
            'model_params': model_params,
            'optim_params': optim_params
        }, '{}/chkpt_{}.pt'.format(dir_chkpt, epoch))
        return '{}/chkpt_{}.pt'.format(dir_chkpt, epoch)

    def log_text(self, split, text):
        with open('{}/log_{}.txt'.format(self.dir_name, split), 'a+') as f:
            f.writelines(text)
    def log(self, epoch, metric, name, split='', avg=True):
        for metric_name, meter in metric.metrics.items():
            if avg:
                value = meter.avg
            else:
                value = meter.val
            self.writer.add_scalar('{}/{}/{}'.format(name, metric_name, split), value, epoch)



def read_args():
    parser = argparse.ArgumentParser(description='RL MultiAgent.')

    # Dataset
    parser.add_argument('--dataset_folder', default='dataset_toy4', type=str)  # dataset_subgoals

    # Model params
    parser.add_argument('--action_dim', default=50, type=int)
    parser.add_argument('--object_dim', default=50, type=int)
    parser.add_argument('--relation_dim', default=50, type=int)
    parser.add_argument('--state_dim', default=50, type=int)
    parser.add_argument('--agent_dim', default=50, type=int)
    parser.add_argument('--num_goals', default=3, type=int)

    parser.add_argument('--max_nodes', default=100, type=int)
    parser.add_argument('--max_edges', default=700, type=int)
    parser.add_argument('--max_steps', default=10, type=int)
    parser.add_argument('--pomdp', action='store_true')  # whether to use the true state or the test state
    parser.add_argument('--graphsteps', default=3, type=int)
    parser.add_argument('--invertedge', action='store_true')


    # Training params
    parser.add_argument('--num_rollouts', default=5, type=int)
    parser.add_argument('--num_epochs', default=150, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--num_workers', default=30, type=int)
    parser.add_argument('--training_mode', default='bc', type=str, choices=['bc', 'pg'])
    parser.add_argument('--lr', default=1.e-3, type=float)

    parser.add_argument('--weights', default='', type=str)


    # RL params
    parser.add_argument('--eps_greedy', default=0., type=float)
    parser.add_argument('--off_policy', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.)
    parser.add_argument('--envstop', action='store_true')

    # Logging
    parser.add_argument('--log_dir', default='logdir', type=str)
    parser.add_argument('--print_freq', default=20, type=int)
    parser.add_argument('--save_freq', default=2, type=int)

    # Running mode
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--dotest', action='store_true')
    parser.add_argument('--interactive', action='store_true')
    parser.add_argument('--eval', action='store_true')

    # Chkpts
    parser.add_argument('--load_path', default=None, type=str)
    parser.add_argument('--continueexec', action='store_true')

    args = parser.parse_args()
    return args

def setup(path_name=None):
    args = read_args()
    helper = Helper(args, path_name)
    return helper

def pretty_instr(instr):
    action, o1, o2 = instr
    o1s = '<{}> ({})'.format(o1[0], o1[1]) if o1 is not None and o1[0] not in ['other', 'no_obj', 'stop'] else ''
    o2s = '<{}> ({})'.format(o2[0], o2[1]) if o2 is not None and o2[0] not in ['other', 'no_obj', 'stop'] else ''
    instr_str = '[{}] {} {}'.format(action, o1s, o2s)
    return instr_str

def pretty_print_program(program, other=None):

    finished = [False]
    program_joint = list(zip(*program))
    # final_instr = [it for it, x in enumerate(program_joint) if x[0] == 'stop']

    if other is not None:
        finished.append(False)
        program_joint2 = list(zip(*other))
        # final_instr2 = [it for it, x in enumerate(program_joint2) if x[0] == 'stop']

    else:
        program_joint2 = [None]*len(program_joint)

    if len(program_joint) > len(program_joint2):
        program_joint2 += [None]*(len(program_joint)-len(program_joint2))
    else:
        program_joint += [None]*(len(program_joint2)-len(program_joint))

    #if len(final_instr) > 0:
    #    program_joint = program_joint[:final_instr[0]]

    instructions = []
    instructions.append('{:70s} | {}'.format('PRED', 'GT'))
    for instr, instr2 in zip(program_joint, program_joint2):
        instr_str = ''
        if instr is not None:
            instr_str = pretty_instr(instr)

        if instr2 is not None:
            instr_str2 = pretty_instr(instr2)
            instr_str = '{:70s} | {}'.format(instr_str, instr_str2)
        instructions.append(instr_str)
    return '\n'.join(instructions)


def get_program_from_nodes(dataset, object_names, object_ids, program):

    object_names_1 = object_names[np.arange(object_names.shape[0])[:, None],
                                     np.arange(object_names.shape[1])[None, :], program[1]]
    object_ids_1 = object_ids[np.arange(object_names.shape[0])[:, None],
                                 np.arange(object_names.shape[1])[None, :], program[1]]
    object_names_2 = object_names[np.arange(object_names.shape[0])[:, None],
                                     np.arange(object_names.shape[1])[None, :], program[2]]
    object_ids_2 = object_ids[np.arange(object_names.shape[0])[:, None],
                                 np.arange(object_names.shape[1])[None, :], program[2]]
    action = program[0]

    instr = obtain_list_instr(action, object_names_1, object_ids_1, object_names_2, object_ids_2, dataset)
    return instr


def obtain_list_instr(actions, o1_names, o1_ids, o2_names, o2_ids, dataset):
    # Split by batch
    actions = torch.unbind(actions.cpu().data, 0)
    o1_names = torch.unbind(o1_names.cpu().data, 0)
    o1_ids = torch.unbind(o1_ids.cpu().data, 0)
    o2_names = torch.unbind(o2_names.cpu().data, 0)
    o2_ids = torch.unbind(o2_ids.cpu().data, 0)

    num_batches = len(actions)
    programs = []
    for it in range(num_batches):
        o1 = zip(list(o1_names[it].numpy()), list(o1_ids[it].numpy()))
        o2 = zip(list(o2_names[it].numpy()), list(o2_ids[it].numpy()))

        action_list = [dataset.action_dict.get_el(x) for x in list(actions[it].numpy())]
        object_1_list = [(dataset.object_dict.get_el(x), idi) for x, idi in o1]
        object_2_list = [(dataset.object_dict.get_el(x), idi) for x, idi in o2]


        programs.append((action_list, object_1_list, object_2_list))
    return programs



class AvgMetrics:
    def __init__(self, metric_list, fmt=':f'):
        self.metric_list = metric_list
        self.metrics = {x: AverageMeter(x, fmt) for x in metric_list}

    def reset(self):
        for _, metric in self.metrics.items():
            metric.reset()

    def update(self, dict_update):
        for metric_name, elem in dict_update.items():
            self.metrics[metric_name].update(elem)

    def __str__(self):
        return '  '.join([str(self.metrics[name]) for name in self.metrics])


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)