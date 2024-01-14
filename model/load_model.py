import numpy as np
from anytown import ChebNet as Anytown_Ori_GCN
from ctown import ChebNet as Ctown_Ori_GCN
from richmond import ChebNet as Richmond_Ori_GCN
from gnn_model import *


def load_model(args, trn_x, trn_y):
    # the original model
    if args.model == 'ori':
        if args.wds == 'anytown':
            model = Anytown_Ori_GCN(np.shape(trn_x)[-1], np.shape(trn_y)[-1])
        elif args.wds == 'ctown':
            model = Ctown_Ori_GCN(np.shape(trn_x)[-1], np.shape(trn_y)[-1])
        elif args.wds == 'richmond':
            model = Richmond_Ori_GCN(np.shape(trn_x)[-1], np.shape(trn_y)[-1])
        else:
            print('Water distribution system is unknown.\n')
            raise Exception('Water distribution system is unknown.\n')

    return model