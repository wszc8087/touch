import sys;

sys.path.insert(0, '.')
sys.path.append("D:/codes/touch")
sys.path.append("D:/codes/touch/classification")
import numpy as np
import sys, os, re, time, shutil, math, random, datetime, argparse
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.cluster import KMeans

from ObjectClusterDataset import ObjectClusterDataset
from Realtimedata import sub_tactile_data
from shared import dataset_tools


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Touch-Classification.')
parser.add_argument('--dataset', default='./data/classification/myfmetadata.mat', help="Path to metadata.mat file.")
parser.add_argument('--nframes', type=int, help='Number of input frames [1--8]', default=8)
parser.add_argument('--reset', type=str2bool, nargs='?', const=True, default=False,
                    help="Start from scratch (do not load weights).")
parser.add_argument('--test', type=str2bool, nargs='?', const=True, default=False, help="Just run test and quit.")
parser.add_argument('--large', type=str2bool, nargs='?', const=True, default=False,
                    help="To use the 9x9 filter version.")  # This was used only for filter viz
parser.add_argument('--snapshotDir', default='snapshots', help="Where to store checkpoints during training.")
args = parser.parse_args()

nFrames = args.nframes
epochs = 30
batch_size = 32
workers = 0

metaFile = args.dataset
doFilter = True


class Trainer(object):

    def __init__(self):
        self.init()
        super(Trainer, self).__init__()

    def init(self):
        # Init model

        self.initModel()

    def run(self):
        print('Starting...')

        self.doSink()
        return

    def doSink(self):
        print('Running test...')
        output, prep = self.step(isTrain=False, sinkName='test')

        # print('Running test with clustering...')
        # val_loader_cluster = self.loadDatasets('test', False, True)
        # res['res_cluster'], res['loss_cluster'] = self.step(val_loader_cluster, self.model.epoch, isTrain=False, sinkName='test_cluster')
        #
        # print('--------------\nResults:')
        
        print("output: ", output)

        index = prep.cpu().numpy()
        print("index: ", index[0][0])
        print("prep: ", prep)
        data = sio.loadmat(args.dataset)
        object = data['objects']
        # print(type(object[0][index]))
        print("Recognize result: the object is", object[0][index][0][0])

    def initModel(self):
        cudnn.benchmark = True

        if args.large:
            from ClassificationModelLargeViz import ClassificationModelLargeViz as Model  # for filter viz only
            initShapshot = 'large_viz'
        else:
            from ClassificationModel import ClassificationModel as Model  # the main model
            initShapshot = 'default'

        snapshot = args.snapshotDir
        initShapshot = os.path.join('snapshots', 'classification', '%s_%dx' % (initShapshot, nFrames),
                                    'checkpoint.pth.tar')

        self.model = Model(numClasses=14, sequenceLength=nFrames)
        self.model.epoch = 0
        self.model.bestPrec = -1e20
        if not initShapshot is None:
            state = torch.load(initShapshot)
            assert not state is None, 'Warning: Could not read checkpoint %s!' % initShapshot
            print('Loading checkpoint %s...' % (initShapshot))
            self.model.importState(state)

    def step(self, isTrain=True, sinkName=None):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top3 = AverageMeter()

        results = {
            'batch': [],
            'rec': [],
            'frame': [],
        }
        catRes = lambda res, key: res[key].cpu().numpy() if not key in results else np.concatenate(
            (results[key], res[key].cpu().numpy()), axis=0)

        end = time.time()

        data_time.update(time.time() - end)

        sequenceLength = args.nframes
        # 测试数据
        # p = list()
        # for i in range(200):
        #     p.append(np.random.randint(500, 650, size=[3, 16, 16]))
        # pressure = np.array(p)

        # 传感器订阅数据
        pressure = sub_tactile_data()

        indices = np.zeros((1, sequenceLength, 3, 16, 16), int)

        # # 方案2：聚类选择N帧
        # channels, row, col = pressure.shape[1], pressure.shape[2], pressure.shape[3]
        # pressure = pressure.reshape(pressure.shape[0], -1)
        # k_means = KMeans(n_clusters=sequenceLength)  # 7个聚类中心
        # result = k_means.fit(pressure)  # Kmeans自动分类
        # kc = result.cluster_centers_  # 自动分类后的聚类中心
        # kc.shape = (sequenceLength, channels, row, col)
        # for i in range(sequenceLength):
        #     indices[0][i] = kc[i]

        # 方案1：随机选择N帧
        index = np.random.choice(np.arange(200), sequenceLength, replace=False)
        for i in range(sequenceLength):
            indices[0][i] = pressure[index[i]]

        indices = np.clip((indices.astype(np.float32)-0)/(256-0), 0.0, 1.0)
        indices = torch.from_numpy(indices)

        inputsDict = {
            'pressure': indices,
        }
        print(inputsDict['pressure'].shape)

        output, prep = self.model.step2(inputsDict, isTrain, params={'debug': True})


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        return output, prep

    @staticmethod
    def make():

        random.seed(454878 + time.time() + os.getpid())
        np.random.seed(int(12683 + time.time() + os.getpid()))
        torch.manual_seed(23142 + time.time() + os.getpid())

        ex = Trainer()
        ex.run()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
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


if __name__ == "__main__":
    Trainer.make()
