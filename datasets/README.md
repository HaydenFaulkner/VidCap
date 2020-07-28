<h1 align='center'>Datasets</h1>
The currently supported datasets are - MSVD and MSRVTT.

<h2 align='center'></h2>
<h2 align='center'>Organisation</h2>

<p align="center">The datasets should be stored in the following directory structure</p>
<pre>
VidCap/
└── datasets/
    ├── ActivityNet
    ├── Flickr30k
    ├── MSCoco
    ├── MSRVTT
    ├── MSVD
    ├── PascalVOC
    └── # version controlled files
</pre>

<h2 align='center'></h2>
<h2 align='center'>Downloading</h2>
All datasets can be downloaded from my <a href="https://drive.google.com/drive/folders/1x79iF5-pRow7i5-R4qX09XEdN-VOgV5e?usp=sharing">Google Drive</a>:
<ul>
    <li><a href="https://drive.google.com/drive/folders/1Y3K6tWtRSM3LiadXRTsZuBOPULccIovf?usp=sharing">PascalVOC (07 + 12)</a></li>
    <li><a href="https://drive.google.com/drive/folders/1wU6rzc7Qv1kB1LwA5XCsk8tlWICNrQiU?usp=sharing">Flickr30k</a></li>
    <li><a href="https://drive.google.com/drive/folders/1xIsUUwSIABrI5yhrTVB4P248ysghtq4t?usp=sharing">MSCoco</a></li>
    <li><a href="https://drive.google.com/drive/folders/1Xdt1Im4IEfuWq-404xBV2h-Sq1J4dSHA?usp=sharing">MSVD</a></li>
    <li><a href="https://drive.google.com/drive/folders/1VUhVWhPB8NU3My4biMqQy2ff4vQEUvo3?usp=sharing">MSRVTT</a></li>
    <li><a href="https://drive.google.com/drive/folders/1VwiCrZZa6d_oj_FM4ZmQnXT0-5sHkYOQ?usp=sharing">ActivityNet (Original + Entities + Captions)</a></li>
</ul>

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