# The Neural Network model in the paper titled "Grounding Gradable Adjectives through Crowdsourcing". Submitted to  LREC 2018.


## Reproducibility

This repository contains the files necessary to reproduce the Neural Network model.

Rather than providing specific hyper parameters values and requiring the model to be
retrained (which usually takes around 12 hours), the repository also contains a PyTorch
model trained as part of the submission.

The submission can easily be reproduced by loading this model using the
`initializer.py` script and following the instructions on screen.

## Getting started

To get started, simply download the files in this repository to a local
directory.

### Prerequisites
The model was developed, trained and tested using the
following packages on a Mac OSX:10.12.6 Sierra

A detailed list of packages are given at the end of this document.
Following the list of instructions in the `Steps to Install` section should get all these packages installed.

```
Conda==4.3.30
Python==3.6.3
PyTorch=0.1.12
NumPy==1.13.1
Pandas=0.21.0
Pip=9.0.1
Scipy=0.19.1
Tqdm=4.19.4
```




Please note that compatibility of the saved model with newer versions
of `PyTorch` has not been checked. Accordingly, please use the
`PyTorch` version listed above.
### Installing

Other than ensuring the dependencies are in place, no separate
installation is required.

Simply execute the `initializer.py` file once the repository has been
saved locally.

## Steps to Install

Please follow the following steps in order:
```
conda create --name grounding python==3.6.3 scipy==0.19.1 pandas==0.21.0
source activate grounding
pip install sklearn
conda install pytorch torchvision -c pytorch
pip install tqdm
pip install torchwordemb
```

## Steps to run
If you are not already in the home folder:
```
cd src/main/python

python initializer.py
```


## OnScreen Instructions:
There are 7 modes in which this code can be run. Below are the details of various modes you are asked to choose from in the welcome screen.

### To train and save using adj based split press :1

In this mode the data that was gained from crowd sourcing experiment is first grouped based on the adjectives. Then the data is split into training-dev-test partitions, in 80-10-10 splits.

### To train on all the data (not adj based split) press :2
In this mode the data that was gained from crowd sourcing experiment is used as is. i.e it is NOT grouped based on the adjectives. The data is split into training-dev-test partitions, in 80-10-10 splits.


To train with nfoldCV on entire data (no adj based split)  press:3
To train with nfoldCV on  adj based split)  press:4
To test using a saved model on alldata_test_partition which was trained on entire data 80-10-10 press:5
To test using a saved model on adj_based_data_test_partition which was trained on adj_based_split press:6
To test using a saved model on adj_based_data_test_partition which was trained on nfcv 4 chunks press:7
To exit Press:0


##  These are the complete list of packages you need to run this code:
#

```


ca-certificates           2017.08.26           ha1e5d58_0
certifi                   2018.1.18                py36_0
cffi                      1.11.5           py36h342bebf_0
freetype                  2.8                  h12048fb_1
intel-openmp              2018.0.0             h8158457_8
jpeg                      9b                   he5867d9_2
libcxx                    4.0.1                h579ed51_0
libcxxabi                 4.0.1                hebd6815_0
libedit                   3.1                  hb4e282d_0
libffi                    3.2.1                h475c297_4
libgfortran               3.0.1                h93005f0_2
libpng                    1.6.34               he12f830_0
libtiff                   4.0.9                h0dac147_0
mkl                       2018.0.1             hfbd8650_4
ncurses                   6.0                  hd04f020_2
numpy                     1.14.1           py36ha726252_2
olefile                   0.45.1                   py36_0
openssl                   1.0.2n               hdbc3d79_0
pandas                    0.21.0           py36hfed917e_1
pillow                    5.0.0            py36hfcce615_0
pip                       9.0.1                    py36_5
pycparser                 2.18             py36h724b2fc_1
python                    3.6.3                h47c878a_7
python-dateutil           2.6.1            py36h86d2abb_1
pytorch                   0.3.1           py36_cuda0.0_cudnn0.0_2    pytorch
pytz                      2018.3                   py36_0
readline                  7.0                  hc1231fa_4
scikit-learn              0.19.1                    <pip>
scipy                     0.19.1           py36h3e758e1_3
setuptools                38.5.1                   py36_0
six                       1.11.0           py36h0e22d5e_1
sklearn                   0.0                       <pip>
sqlite                    3.22.0               h3efe00b_0
tk                        8.6.7                h35a86e2_3
torchvision               0.2.0            py36hf5eb7ec_1    pytorch
torchwordemb              0.0.8                     <pip>
tqdm                      4.19.6                    <pip>
wheel                     0.30.0           py36h5eb2c71_1
xz                        5.2.3                h0278029_2
zlib                      1.2.11               hf3cbc9b_2

```