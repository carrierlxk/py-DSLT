:tada:

Pytorch version is release:
--
Requirement
--
* Python 2.7 (I use Anaconda 2.* here. If you use Python3, you may get the very different results!)
* Python-opencv
* PyTorch 0.40
* other common packages such as numpy, etc


## Data Preparison

* Download ILSVRC15, and unzip it (let's assume that $ILSVRC2015_Root is the path to your ILSVRC2015)

  * Move $ILSVRC2015_Root/Data/VID/val into $ILSVRC2015_Root/Data/VID/train/, so we have five sub-folders in $ILSVRC2015_Root/Data/VID/train/
  * It is a good idea to change the names of five sub-folders in $ILSVRC2015_Root/Data/VID/train/ to a, b, c, d, and e
Move $ILSVRC2015_Root/Annotations/VID/val into $ILSVRC2015_Root/Annotations/VID/train/, so we have five sub-folders in $ILSVRC2015_Root/Annotations/VID/train/
  * Change the names of five sub-folders in $ILSVRC2015_Root/Annotations/VID/train/ to a, b, c, d and e, respectively
* Generate image crops
  * cd $SiamFC-PyTorch/ILSVRC15-curation/ (Assume you've downloaded the rep and its path is $SiamFC-PyTorch)
  * change vid_curated_path in gen_image_crops_VID.py to save your crops
  * run $python gen_image_crops_VID.py (I run it in PyCharm), then you can check the cropped images in your saving path (i.e., vid_curated_path)
* Generate imdb for training and validation
  * cd $SiamFC-PyTorch/ILSVRC15-curation/
  * change vid_root_path and vid_curated_path to your custom path in gen_imdb_VID.py
  * run $python gen_imdb_VID.py, then you will get two json files imdb_video_train.json (~ 430MB) and imdb_video_val.json (~ 28MB) in current folder, which are used for training and validation

---
## Train
* cd $SiamFC-PyTorch/Train/
* Change data_dir, train_imdb and val_imdb to your custom cropping path, training and validation json files
* run $python run_Train_SiamFC.py
* some notes in training 
 * the parameters for training are in Config.py
 * by default, I use GPU in training, and you can check the details in the function train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True)
 * by default, the trained models will be saved to $SiamFC-PyTorch/Train/model/
 
 ---
 ## Test
* cd $SiamFC-PyTorch/Tracking/
* Firstly, you should take a look at Config.py, which contains all parameters for tracking
* Change self.net_base_path to the path saving your trained models
* Change self.seq_base_path to the path storing your test sequences (OTB format, otherwise you need to revise the function load_sequence() in Tracking_Utils.py
* Change self.net to indicate whcih model you want for evaluation (by default, use the last one), and I've uploaded a trained model SiamFC_50_model.pth in this rep (located in $SiamFC-PyTorch/Train/model/)
