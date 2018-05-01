#!/usr/bin/env python
from __future__ import print_function
import sklearn as sklearn
from sklearn.metrics import r2_score
#import matplotlib.pyplot as plt
from scipy import stats
from itertools import count
import torch.nn as nn
import torch
import torch.autograd
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import sys
import scipy as scipy
from tqdm import tqdm
POLY_DEGREE = 1
##W_target = torch.randn(POLY_DEGREE, 1) * 5
#b_target = torch.randn(1) * 5
import torch.optim as optim
#
# def make_features(x):
#     """Builds features i.e. a matrix with columns [x, x^2, x^3, x^4]."""
#     x = x.unsqueeze(1)
#     return torch.cat([x ** i for i in range(1, POLY_DEGREE+1)], 1)
#

# def f(x):
#     """Approximated function."""
#     return x.mm(W_target) + b_target[0]

def check_vectors_have_same_dimensions(Y,Y_):
    '''
    Checks that vector Y and Y_ have the same dimensions. If they don't
    then there might be an error that could be caused due to wrong broadcasting.
    '''
    DY = tuple( Y.size() )
    DY_ = tuple( Y_.size() )
    if len(DY) != len(DY_):
        return True
    for i in range(len(DY)):
        if DY[i] != DY_[i]:
            return True
    return False


def poly_desc(W, b):
    """Creates a string description of a polynomial."""
    result = 'y = '
    for i, w in enumerate(W):
        result += '{:+.2f} x '.format(w)
    result += '{:+.2f}'.format(b[0])
    return result



def convert_variable(features, labels):
    """Builds a batch i.e. (x, f(x)) pair."""

    #the actual features and y comes here.
    # Toy data
    # x = np.random.randn(3, 3).astype('f')
    # x2=torch.from_numpy(x)
    # w = np.array([[1], [2], [3]],dtype="float32")
    # y = np.dot(x, w)
    # y2 = torch.from_numpy(y)

    #actual data


    x2 =torch.from_numpy(features)
    y2 = torch.from_numpy(labels)

    # print("x2")
    # print(x2)
    # print("y2")
    # print(y2)

    return Variable(x2), Variable(y2,requires_grad=False)

def runLR(features, y):
    featureShape=features.shape
    # print("Features row:" + str(features[0]))
    # # print("y row:" + str(y[0]))
    #
    # print("featureShape")
    # print(featureShape)
    # print("size of big y is:")
    # print((y.shape))


    # features = features[:20]
    # y = y[:20]

    # create teh weight matrix. the dimensions must be transpose
    # of your features, since they are going to be dot producted
    fc = torch.nn.Linear(featureShape[1],1)#, bias=False)




    pred_y = None
    noOfEpochs=10000

    for epoch in tqdm(range(noOfEpochs),total=noOfEpochs,desc="epochs:"):

        # Reset gradients
        fc.zero_grad()

        #np.random.shuffle(features)
        # Get data
        batch_x, batch_y = convert_variable(features, y)

        #self.hidden_dim = hidden_dim

        loss_fn = nn.MSELoss(size_average=True)
        optimizer = optim.SGD(fc.parameters(), lr=0.00000001)
        adam = optim.Adam(fc.parameters())
        rms = optim.RMSprop(fc.parameters(),lr=1e-5, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0)

        #multiply weight with input vector
        affine=fc(batch_x)
        pred_y=affine.data.cpu().numpy()


        loss = loss_fn(affine, batch_y)


        # Backward pass
        loss.backward()

        # optimizer.step()
        # adam.step()
        rms.step()

        # # Apply gradients
        # for param in fc.parameters():
        #     param.data -= 0.001 * param.grad.data

        # for param in fc.parameters():
        #     param.data.add_(-0.1 * param.grad.data)


        # print(loss)


        # # Stop criterion
        # if loss < 1e-3:
        #     break

    print('Loss: after all epochs'+str((loss.data)))

    #print("y value:")
    #print(y)
    #print("predicted y value")
    #print(pred_y)
    rsquared_value=r2_score(y, pred_y, sample_weight=None, multioutput='uniform_average')


    print("rsquared_value:")
    print(str(rsquared_value))

    # #rsquared_value2= rsquared(y, pred_y)
    # print("rsquared_value2:")
    # print(str(rsquared_value2))

    #print(fc.weight.data.view(-1))
    learned_weights = fc.weight.data
    return(learned_weights.cpu().numpy())
   # print('==> Learned function:\t' + poly_desc(fc.weight.data.view(-1), fc.bias.data))
    #print('==> Actual function:\t' + (W_target.view(-1), b_target))


def rsquared(x, y):
    """ Return R^2 where x and y are array-like."""

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)

    #to plot#
    # plt.plot(x, y, 'o', label='original data')
    # plt.plot(x, intercept + slope*x, 'r', label='fitted line')
    # plt.legend()
    # plt.show()

    return r_value**2
