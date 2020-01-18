# Check Out the weights path from Keras_frcnn.Config file last line and parser paths for each of the train_rpn.py, train_frcnn.py and test_frcnn.py
# We have already provided default paths and other values so please check it out ones before executing. Code is 100% working

# What is this repo?
- **Simple faster-RCNN codes in Keras!**

- **RPN (region proposal layer) can be trained separately!**

- **Active support! :)**

- **MobileNet support!**

- **VGG support!**


## Compared to the forked keras-frcnn..
1. mobilenetv1 and mobilenetv2(TBD) support added (partially). Can also try Mobilenetv1_05,Mobilenetv1_25 for smaller nets.
2. VGG19 support added.
3. RPN can be trained seperately.


# Running scripts..

## 1. clone the repo

``` 
git clone https://github.com/kentaroy47/frcnn-from-scratch-with-keras.git
cd frcnn-from-scratch-with-keras
```

Install requirements. make sure that you have Keras installed.
```
pip install -r requirements.txt
```

## 2. Download pretrained weights.
Using imagenet pretrained VGG16 weights will significantly speed up training.

Download and place it in the root directory.

You can choose other base models as well.

```
# for VGG16
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

# for mobilenetv1
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.6/mobilenet_1_0_224_tf.h5

# for resnet 50
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/resnet50_weights_tf_dim_ordering_tf_kernels.h5
```

Other tensorflow pretrained models are in bellow.

https://github.com/fchollet/deep-learning-models/releases/


## 3. lets train region proposal network first, rather than training the whole network.
Training the entire faster-rcnn is quite difficult, but RPN itself can be more handy!

You can see if the loss converges.. etc

```
python train_rpn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/

Epoch 1/20
100/100 [==============================] - 57s 574ms/step - loss: 5.2831 - rpn_out_class_loss: 4.8526 - rpn_out_regress_loss: 0.4305 - val_loss: 4.2840 - val_rpn_out_class_loss: 3.8344 - val_rpn_out_regress_loss: 0.4496
Epoch 2/20
100/100 [==============================] - 51s 511ms/step - loss: 4.1171 - rpn_out_class_loss: 3.7523 - rpn_out_regress_loss: 0.3649 - val_loss: 4.5257 - val_rpn_out_class_loss: 4.1379 - val_rpn_out_regress_loss: 0.3877
Epoch 3/20
100/100 [==============================] - 49s 493ms/step - loss: 3.4928 - rpn_out_class_loss: 3.1787 - rpn_out_regress_loss: 0.3142 - val_loss: 2.9241 - val_rpn_out_class_loss: 2.5502 - val_rpn_out_regress_loss: 0.3739
Epoch 4/20
 80/100 [=======================>......] - ETA: 9s - loss: 2.8467 - rpn_out_class_loss: 2.5729 - rpn_out_regress_loss: 0.2738  

```

## 4. then train the whole Faster-RCNN network!

```
python train_.frcnn.py --network mobilenetv1 -o simple -p /path/to/your/dataset/

Using TensorFlow backend.
Parsing annotation files
Training images per class:
{*, 'bg': 0}
Num classes (including bg) = 2
Config has been written to config.pickle, and can be loaded when testing to ensure correct results
Num train samples 401
Num val samples 88
loading weights from ./pretrain/mobilenet_1_0_224_tf.h5
loading previous rpn model..
no previous model was loaded
Starting training
Epoch 1/200
100/100 [==============================] - 150s 2s/step - rpn_cls: 4.5333 - rpn_regr: 0.4783 - detector_cls: 1.2654 - detector_regr: 0.1691  
Mean number of bounding boxes from RPN overlapping ground truth boxes: 1.74
Classifier accuracy for bounding boxes from RPN: 0.935625
Loss RPN classifier: 4.244322432279587
Loss RPN regression: 0.4736669697239995
Loss Detector classifier: 1.1491613787412644
Loss Detector regression: 0.20629869312047958
Elapsed time: 150.15273475646973
Total loss decreased from inf to 6.07344947386533, saving weights
Epoch 2/200
Average number of overlapping bounding boxes from RPN = 1.74 for 100 previous iterations
 38/100 [==========>...................] - ETA: 1:24 - rpn_cls: 3.2813 - rpn_regr: 0.4576 - detector_cls: 0.8776 - detector_regr: 0.1826

```

# Dataset setup.
You can either try voc or simple parsers for your dataset.

simple parsers are much easier, while you train your network as:

```
python train_rpn.py --network vgg16 -o simple -p ./dataset.txt
```

Simply provide a text file, with each line containing:
```
filepath,x1,y1,x2,y2,class_name
```
For example:
```dataset.txt
/data/imgs/img_001.jpg,837,346,981,456,* /data/imgs/img_002.jpg,215,312,279,391,*

Note : Here * means class name which includes both Pedestrain and Cyclist.


```
#------------------------------------------HAPPY----Hacking-------------------------------------------------------------#