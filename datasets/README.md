<h1 align='center'>Datasets</h1>

<h2 align='center'></h2>
<h2 align='center'>Downloading</h2>
<h3 align='center'>Pascal VOC (Image Captioning)</h3>
<p align="center">Consists of 1000 images from <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit">PascalVOC</a> with five 
captions per image. The dataset is presented <a href="http://vision.cs.uiuc.edu/pascal-sentences/">here</a></p>

<p align="center">To download run <a href="get_voc_dataset.sh"><code>get_voc_dataset.sh</code></a> from the root dir:</p>

```
VidCap$ python datasets/get_voc_dataset.sh
```

<p align="center">The script will download the <a href="http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devki">Pascal VOC</a> dataset and 
put the sentences into:</p>

```
datasets/PascalVOC/VOCdevkit/Sentences
```

<p align="center">.......</p>
<h3 align='center'>Flickr 30k (Image Captioning)</h3>

<p align="center">The dataset is available for <a href="https://www.kaggle.com/hsankesara/flickr-image-dataset/downloads/flickr-image-dataset.zip/1">download from 
Kaggle</a> (requires sign in). 
Or the official website is <a href="http://hockenmaier.cs.illinois.edu/DenotationGraph/">here</a>, but you will need to fill out a 
form due to the copyright of the Flickr images, and the data <em>might</em> get sent to you</p>

<p align="center">To organise the dataset make a <code>Flickr30k</code> directory in <code>datasets</code>:</p>

```
cd datasets
mkdir Flickr30k
```

<p align="center"> and place the downloaded <code>flickr-image-dataset.zip</code> in it, resulting in:</p>

```commandline
datasets/Flickr30k/flickr-image-dataset.zip
```

<p align="center">Then run <a href="organise_flickr30k.sh"><code>organise_flickr30k.sh</code></a> from the root dir:</p>

```
VidCap$ . datasets/organise_flickr30k.sh
```

<p align="center">If you also want <a href="https://github.com/BryanPlummer/flickr30k_entities">Flickr 30k Entities</a>, which adds 244k 
coreference chains and 276k manually annotated bounding boxes, then follow up by running <a href="get_flickr30k_entities.sh"><code>get_flickr30k_entities.sh</code></a>:</p>

```
VidCap$ . datasets/get_flickr30k_entities.sh
```

<p align="center">.......</p>
<h3 align='center'>MS Coco (Image Captioning)</h3>
<p align="center">To download run <a href="get_coco_dataset.sh"><code>get_coco_dataset.sh</code></a> from the root dir:</p>

```
VidCap$ . datasets/get_coco_dataset.sh
```

<p align="center">.......</p>
<h3 align='center'>ActivityNet (Video Captioning)</h3>

<p align="center">To download run <a href="get_activitynet_dataset.sh"><code>get_activitynet_dataset.sh</code></a> from the root dir:</p>

```
VidCap$ . datasets/get_activitynet_dataset.sh
```

<p align="center">This script will also attempt to download the videos from YouTube, note this can take a very long time and also that not
 all videos are still on YouTube. To get the full dataset instead you can fill out <a href="https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform">this form</a>.</p>
 
<p align="center">.......</p>
<h3 align='center'>MSVD (Video Captioning)</h3>

<p align="center">Manually download the dataset from <a href="https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAC_TAsR7SkEYhkSdAFKcBlMa?dl=0">dropbox</a>
 and save in a <code>MSVD</code> directory as <code>naacl15_translating_videos_processed_data.zip</code>:</p>
 
```
datasets/MSVD/naacl15_translating_videos_processed_data.zip
```

<p align="center">Then run the <a href="get_msvd_dataset.sh"><code>get_msvd_dataset.sh</code></a> from the root dir:</p>

```
VidCap$ . datasets/get_msvd_dataset.sh
```

<p align="center">.......</p>
<h3 align='center'>MSR-VTT (Video Captioning)</h3>
 
<p align="center">To download the training <code>.json</code> run <a href="get_msrvtt_dataset.sh"><code>get_msrvtt_dataset.sh</code></a> from the root dir:</p>

```
VidCap$ . datasets/get_msrvtt_dataset.sh
```

<p align="center">To get the videos you can use the mediafire.com (<em>ew</em>) links below at your own risk:</p>
<ul>
<!-- <li><a href="http://download1515.mediafire.com/t1cfuz3q7tdg/s88kuv5kqywpyym/train_val_annotation.zip">train_val_annotation.zip</a></li> -->
<!-- <li><a href="http://download847.mediafire.com/egekeag8fowg/wvw68y9wmo3iw80/test_videodatainfo.json.zip">test_videodatainfo.json.zip</a></li> -->
<li><a href="http://download1079.mediafire.com/2xemo9i5s5jg/x3rrbe4hwp04e6w/train_val_videos.zip">train_val_videos.zip</a></li>
<li><a href="http://download876.mediafire.com/yf43j27femyg/czh8sezbo9s4692/test_videos.zip">test_videos.zip</a></li>
</ul>
<p align="center">Once downloaded extract the zips and move all the videos from their split directories directly into a <code>/datasets/MSRVTT/videos/</code> directory.</p>

<p align="center">Instead of this you can just instantiate an <code>MSRVTT()</code> instance with <code>download_missing=True</code>, which will try and download the videos from youtube. Note not all may be available.</p>

<h2 align='center'></h2>
<h2 align='center'>Usage</h2>
<p align="center"><b>TODO: update for Image Captioning Sets</b></p>

<p align="center">Datasets can be initialised with their corresponding class (for example for MSVD):</p>

```python
train_dataset = MSVD()
```

<p align="center">This will default to the <b>training split</b>, a different split can be specified with the <code>splits</code> argument, which takes as list of strings:</p>

```python
val_dataset = MSVD(splits=['val'])
trainval_dataset = MSVD(splits=['train', 'val'])
```

<p align="center">Furthermore a <code>.tree</code> file can be passed in with the <code>subset</code> argument to only include samples that contain an instance of a tree object in the caption:</p>

```python
train_dataset = MSVD(subset='filtered_det.tree')
```

<h2 align='center'></h2>
<h2 align='center'>Stats</h2>

<p align="center">The following stats can be printed via each dataset instances <code>stats()</code> function:</p>

<p align="center">.......</p>
<h3 align='center'>MSVD</h3>

<p align="center">Train Split</p>

```
# Images: 1200
# Captions: 48774
# Words: 342767
# Nouns 119092 (34% of words)
# Verbs 53050 (15% of words)
Vocab: 9383
Nouns Vocab 5992 (63% of Vocab)
Verbs Vocab 3204 (34% of Vocab)

Captions per image (min, avg, max): 18, 40, 66
Words per image (min, avg, max): 89, 285, 555
Nouns per image (min, avg, max): 27, 99, 206
Verbs per image (min, avg, max): 16, 44, 120

Vocab (unique words) per image (min, avg, max): 22, 60, 125
Nouns Vocab (unique words) per image (min, avg, max): 5, 24, 59
Verbs Vocab (unique words) per image (min, avg, max): 2, 15, 42
```


<p align="center">.......</p>
<h3 align='center'>MSR-VTT</h3>

<p align="center">Train Split</p>

```
# Images: 10000
# Captions: 200000
# Words: 1854960
# Nouns 666133 (35% of words)
# Verbs 255301 (13% of words)
Vocab: 28464
Nouns Vocab 19712 (69% of Vocab)
Verbs Vocab 8973 (31% of Vocab)

Captions per image (min, avg, max): 20, 20, 20
Words per image (min, avg, max): 92, 185, 419
Nouns per image (min, avg, max): 26, 66, 134
Verbs per image (min, avg, max): 3, 25, 62

Vocab (unique words) per image (min, avg, max): 13, 70, 154
Nouns Vocab (unique words) per image (min, avg, max): 6, 29, 76
Verbs Vocab (unique words) per image (min, avg, max): 1, 15, 42
```