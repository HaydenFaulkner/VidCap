<h1 align='center'>Feature Generation</h1>
<p align="center">The captioning pipeline is dependent on a bunch of image and/or video features. For computational efficiency these features are pre-extracted with the set of scripts in this directory.</p>

<h2 align='center'></h2>
<h2 align='center'>Video Features</h2>
<h3 align='center'>Global Video Features</h3>
<p align="center">When working with videos it is desirable to utilise features that are representative of the video as a whole - global features.</p>

<p align="center">To generate global features using video classification pipelines you can use the <a href="global/vid/feat_extract.py">feat_extract.py</a> script:</p>

```cmd
path/to/VidCap$ python feature_gen/global/vid/feat_extract.py --dataset MSVD --model i3d_resnet50_v1_kinetics400 
```
<p align="center">If you want a stronger feature by covering more temporal information you can use the <code>--num-segments</code> argument to extract features from n segments of the video and combine them:</p>

```cmd
path/to/VidCap$ python feature_gen/global/vid/feat_extract.py --dataset MSVD --model i3d_resnet50_v1_kinetics400 --num-segments 10
```
<p align="center">If you want to change the temporal length of each segment you can use the <code>--new-length</code> argument to make the segments n frames long:</p>

```cmd
path/to/VidCap$ python feature_gen/global/vid/feat_extract.py --dataset MSVD --model i3d_resnet50_v1_kinetics400 --num-segments 10 --new-length 64
```

<p align="center">If you want to use three crops (either center+left+right or center+top+bottom depending on video orientation) you can use the <code>--three-crop</code> argument:</p>

```cmd
path/to/VidCap$ python feature_gen/global/vid/feat_extract.py --dataset MSVD --model i3d_resnet50_v1_kinetics400 --num-segments 10 --new-length 64 --three-crop
```

<p align="center">You can also use <a href="http://openaccess.thecvf.com/content_ICCV_2019/papers/Feichtenhofer_SlowFast_Networks_for_Video_Recognition_ICCV_2019_paper.pdf">SlowFast networks</a> models with the use of <code>slowfast_4x16_resnet50_kinetics400</code>:</p>

```cmd
path/to/VidCap$ python feature_gen/global/vid/feat_extract.py --dataset MSVD --model slowfast_4x16_resnet50_kinetics400 --slowfast --slow-temporal-stride 16 --fast-temporal-stride 2 
```


<h3 align='center'>.......</h3>
<h3 align='center'>Pre-Extracted</h3>
<p align="center">Below you can download pre-extracted features:</p>

<a href="https://drive.google.com/drive/folders/1yfIAy_BIJTcUm8ktDq_eaNf-8xlU8Uks?usp=sharing">MSVD (19MB)</a> and <a href="https://drive.google.com/drive/folders/1EfU0ZXZyMNNwu6eMhPoKZ3Ev3kp3BmIc?usp=sharing">MSR-VTT (94MB)</a> using:
```cmd
path/to/VidCap$ python feature_gen/global/vid/feat_extract.py --dataset MSVD --model slowfast_4x16_resnet50_kinetics400 --slowfast --slow-temporal-stride 16 --fast-temporal-stride 2 --use-decord --num-segments 5 
```

<h2 align='center'></h2>
<h2 align='center'>Image Features</h2>
<h3 align='center'>todo</h3>
<p align="center"></p>