# Temporal stable ExpandNet
This project extends ExpandNet project (https://github.com/dmarnerides/hdr-expandnet) - net for HDR images reconstruction.  
Using original net model for reconstructing video results in unstable video (flickering occurs).  

Temporal stable ExpandNet project implements solution improving stability of reconstructed videos by adding regularization term to loss function (https://arxiv.org/abs/1902.10424).

## Requirements
Virtual environment usage is recommended.
```
pip install -r requirements.txt
```
Solution works fine with Python 3.8. OpenEXR may should be installed with apt-get: sudo apt-get install openexr and libopenexr-dev on Ubuntu.

## Usage
For training:
```
python train.py -h
```
Training consists of two phases (--train_phase flag). During first phase model should be trained only reconstruction and during second phase model should be trained reconstruction with stability (loss function with two terms).  
  
For testing:
```
python expand_image.py -h
python expand_video.py -h
```
expand_image.py input is set of ldr images, which will be transformed into hdr.
expand_video.py input is video (ldr frames or video file) which will be transformed into hdr video frames.

## Prepared models
As a result of analysis, ready to use models were trained. Those models are stored in 'net_params' directory. Explanation of models names:

Training consists of two phases: base and fine tuning. Model name indicates tactic used during training phase. 
"base" indicates first phase of training, "tuning" second phase.
"Rec" means that net was trained only in terms of reconstruction, "Stab" means that net was trained in terms of reconstruction and stability.
Loss at the end of model name indicates loss function used as stability term. 

- baseRec_tuningRec: base model, trained only in terms of reconstruction
- baseRec_tuningStab_L1Loss: fine tuning with stability term (L1 loss used) 
- baseRec_tuningStab_L2Loss: fine tuning with stability term (L2 loss used)
- baseRec_tuningStab_recLoss: fine tuning with stability term (expandNet loss used)
- baseRec_tuningStab_ssimLoss: fine tuning with stability term (ssim used)

## Useful links
Projects used to compare:
- https://github.com/marcelsan/Deep-HdrReconstruction
- https://github.com/gabrieleilertsen/hdrcnn

Data used for training (license of those data sets allows to use, share, modify etc. - Creative Commons):
- https://polyhaven.com/hdris (possible to download all data with polyhaven api)
- https://hdrplusdata.org/dataset.html (raw photos .dng transformed to .exr with imagemagick - https://imagemagick.org/index.php)  
Training data set is a mix of polyhaven and hdrplus data (480+520 images), validation and test data set consists of hdrplus images.

Hdr online viewer:
- https://viewer.openhdr.org/
