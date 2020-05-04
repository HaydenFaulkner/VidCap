<h1 align='center'>Datasets</h1>

<h2 align='center'>Downloading</h2>
<h3 align='center'>Pascal VOC (Image Captioning)</h3>
Consists of 1000 images from [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) with five 
captions per image. The dataset is presented [here](http://vision.cs.uiuc.edu/pascal-sentences/).

To **download** the dataset run `get_voc_dataset.sh` from the root dir:
```
VidCap$ python datasets/get_voc_dataset.sh
```

The script will download the [PascalVOC](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit) dataset and 
put the sentences into:
```
datasets/PascalVOC/VOCdevkit/Sentences
```

<h3 align='center'>Flickr 30k (Image Captioning)</h3>
Consists of 31783 images with five captions per image (158915 total). The dataset is available for [**download from 
Kaggle**](https://www.kaggle.com/hsankesara/flickr-image-dataset/downloads/flickr-image-dataset.zip/1) (requires sign in). 
Or the official website is [here](http://hockenmaier.cs.illinois.edu/DenotationGraph/), but you will need to fill out a 
form due to the copyright of the Flickr images, and the data **_might_** get sent to you.

To organise the dataset make a `Flickr30k` directory in `datasets`:
```
cd datasets
mkdir Flickr30k
```
and place the downloaded `flickr-image-dataset.zip` in it, resulting in:
```commandline
datasets/Flickr30k/flickr-image-dataset.zip
```

Then run `organise_flickr30k.sh` from the root dir:
```
VidCap$ . datasets/organise_flickr30k.sh
```

If you also want [**Flickr 30k Entities**](https://github.com/BryanPlummer/flickr30k_entities), which adds 244k 
coreference chains and 276k manually annotated bounding boxes, then follow up by running `get_flickr30k_entities.sh`:
```
VidCap$ . datasets/get_flickr30k_entities.sh
```

<h3 align='center'>MS Coco (Image Captioning)</h3>
To **download** the dataset run `get_coco_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_coco_dataset.sh
```

<h3 align='center'>ActivityNet (Video Captioning)</h3>

To **download** the dataset run `get_activitynet_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_activitynet_dataset.sh
```
This script will also attempt to download the videos from YouTube, note this can take a very long time and also that not
 all videos are still on YouTube. To get the full dataset instead you can fill out [**this form**](https://docs.google.com/forms/d/e/1FAIpQLSeKaFq9ZfcmZ7W0B0PbEhfbTHY41GeEgwsa7WobJgGUhn4DTQ/viewform).
 
<h3 align='center'>MSVD (Video Captioning)</h3>

Manually download the dataset from [dropbox](https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAC_TAsR7SkEYhkSdAFKcBlMa?dl=0)
 and save in a `MSVD` directory as `naacl15_translating_videos_processed_data.zip`:
```
datasets/MSVD/naacl15_translating_videos_processed_data.zip
```
Then run the `get_msvd_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_msvd_dataset.sh
```

<h3 align='center'>MSR-VTT (Video Captioning)</h3>
 
To **download** the training `.json` run `get_msrvtt_dataset.sh` from the root dir:
```
VidCap$ . datasets/get_msrvtt_dataset.sh
```

To get the videos you can use the **mediafire.com** (*ew I know*) links below at your own risk:
- [train_val_annotation.zip](http://download1515.mediafire.com/t1cfuz3q7tdg/s88kuv5kqywpyym/train_val_annotation.zip)
- [test_videodatainfo.json.zip](http://download847.mediafire.com/egekeag8fowg/wvw68y9wmo3iw80/test_videodatainfo.json.zip)
- [train_val_videos.zip](http://download1079.mediafire.com/2xemo9i5s5jg/x3rrbe4hwp04e6w/train_val_videos.zip)
- [test_videos.zip](http://download876.mediafire.com/yf43j27femyg/czh8sezbo9s4692/test_videos.zip)

<h2 align='center'>Usage</h2>
<b>TODO: update for Image Captioning Sets</b>

Datasets can be initialised with their corresponding class (for example for MSVD):
```python
train_dataset = MSVD()
```

This will default to the <b>training split</b>, a different split can be specified with the `splits` argument, which takes as `list` of `string`s:
```python
val_dataset = MSVD(splits=['val'])
trainval_dataset = MSVD(splits=['train', 'val'])
```

Furthermore a `.tree` file can be passed in with the `subset` argument to only include samples that contain an instance of a tree object in the caption:
```python
train_dataset = MSVD(subset='filtered_det.tree')
```

<h2 align='center'>Stats</h2>
The following stats can be printed via each dataset instance's `stats()` function

<h3 align='center'>MSVD</h3>
Train Split
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


<h3 align='center'>MSR-VTT</h3>
Train Split
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