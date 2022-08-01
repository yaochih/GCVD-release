# GCVD: Globally Consistent Video Depth and Pose Estimation with Efficient Test-Time Training
This repository contains the pytorch implementations of GCVD.

## Setup
Tested on Python 3.7 and PyTorch 1.8.0 on CUDA 10.1.
```
apt-get install ffmpeg
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.1 -c pytorch
pip install opencv-python scipy tqdm path imageio scikit-image pypng timm tensorboard matplotlib
cd .. # (install detectron2 outside GCVD)
git clone https://github.com/facebookresearch/detectron2 detectron2
pip install -e detectron2
cd GCVD
```
Download [MiDaS](https://github.com/isl-org/MiDaS.git) repo in GCVD
```
git clone https://github.com/isl-org/MiDaS.git
```
Download freeimage reader for handling MiDaS output format (.pfm)
```
python3 -c "import imageio; imageio.plugins.freeimage.download()"
```
Download network weights from [CVD2](https://github.com/facebookresearch/robust_cvd.git) 
the weights include MiDaS single depth, Mask-RCNN, and RAFT flow.
```
wget https://www.dropbox.com/s/avruiwv95c5xucn/models.zip?dl=1 -O weights.zip
unzip weights.zip -d weights
```

### Install g2o
Please follow [g2opy](https://github.com/uoip/g2opy) to install. You may need to install eigen 3.3.4 on your own. \
You may also refer to the issue https://github.com/uoip/g2opy/issues/46#issuecomment-704190419 if you use Ubuntu 20.04.
```
apt-get install build-essential cmake libglu1-mesa-dev libsuitesparse-dev
git clone https://github.com/uoip/g2opy.git
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
cd .. # (back to GCVD root folder)
```
### Running 
Prepare a test video of [7-Scenes dataset](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/) (use `7-Scenes/chess/seq-01` as an example).
```
mkdir test-dataset && cd test-dataset
wget http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/chess.zip -O
unzip chess.zip && cd chess
unzip unzip seq-01.zip
ffmpeg -framerate 25 -pattern_type glob -i "seq-01/*.color.png" -c:v copy seq-01.mkv
cd ../.. # (back to GCVD root folder)
```
Start to run our implementation, it would take some time. \
The results will save in `outputs/test/depths/final` and `outputs/test/camera/final`.
```
python3 main.py test-dataset/chess/seq-01.mkv --name test --pose_graph [(optional) --post_filter]
```
### Visualize the result (optional) 
Install Open3D (tested on 0.13.0)
```
pip install open3d==0.13.0
```
Visualize the result
```
python3 visualize.py outputs/test test-dataset/chess/seq-01
```

---
### Citation
```Bibtex
@InProceedings{
    author    = {Lee, Yao-Chih and Tseng, Kuan-Wei and Chen, Guan-Sheng and Chen, Chu-Song},
    title     = {Globally Consistent Video Depth and Pose Estimation with Efficient Test-Time Training},
    booktitle = {},
    year      = {2022}
}
```
### License
The provided implementation is strictly for academic purposes only.
