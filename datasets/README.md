<h1 align='center'>Datasets</h1>
The currently supported datasets are <a href="https://vsubhashini.github.io/s2vt.html">MSVD</a> and <a href="https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/">MSRVTT</a>.

<h2 align='center'></h2>
<h2 align='center'>Organisation</h2>

<p align="center">The datasets should be downloaded, unzipped and stored in the following directory structure. Clink on the datasets to download <code>.zip</code> files from my Google Drive</p>
<pre>
VidCap/
└── <a href="https://drive.google.com/drive/folders/1x79iF5-pRow7i5-R4qX09XEdN-VOgV5e?usp=sharing">datasets/</a>
    ├── ActivityNet
    ├── <a href="https://drive.google.com/file/d/11CBboQ49VGp1JRgWcxUcRUx-lFW_Xo5Z/view?usp=sharing">Flickr30k</a>
    ├── <a href="https://drive.google.com/file/d/1zzvPAZGonlrdLBXXpKVV9IpbRJxYawLv/view?usp=sharing">MSCoco</a>
    ├── <a href="https://drive.google.com/file/d/1z5cu0y1e36gvlna7Qbb745X7U8LK52pB/view?usp=sharing">MSRVTT</a>
    ├── <a href="https://drive.google.com/file/d/1lM0LAf4lEb2yLIPNGRbIDUtrXp8JdFzF/view?usp=sharing">MSVD</a>
    ├── <a href="https://drive.google.com/file/d/19_HmXUnTwL20ROs-q5e0DUmj3yM5ZXJK/view?usp=sharing">PascalVOC</a>
    └── # version controlled files
</pre>


<h2 align='center'></h2>
<h2 align='center'>Usage</h2>
<p align="center"><b>Image Captioning Sets Coming Soon</b></p>

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
<h2 align='center'>Statistics</h2>

<p align="center">The following stats can be printed via each dataset instances <code>stats()</code> function:</p>

<h3 align='center'>MSVD</h3>

<pre>
---------- Training ----------
# Clips: 1200
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
</pre>
<pre>
---------- Validation ----------
# Clips: 100
# Captions: 4290
# Words: 30210
# Nouns 10371 (34% of words)
# Verbs 4658 (15% of words)
Vocab: 2220
Nouns Vocab 1302 (58% of Vocab)
Verbs Vocab 707 (31% of Vocab)

Captions per image (min, avg, max): 25, 42, 62
Words per image (min, avg, max): 164, 302, 463
Nouns per image (min, avg, max): 52, 103, 151
Verbs per image (min, avg, max): 26, 46, 80

Vocab (unique words) per image (min, avg, max): 23, 61, 106
Nouns Vocab (unique words) per image (min, avg, max): 9, 25, 53
Verbs Vocab (unique words) per image (min, avg, max): 4, 15, 37
</pre>
<pre>
---------- Testing ----------
# Clips: 670
# Captions: 27763
# Words: 194692
# Nouns 67695 (34% of words)
# Verbs 30153 (15% of words)
Vocab: 7006
Nouns Vocab 4376 (62% of Vocab)
Verbs Vocab 2374 (33% of Vocab)

Captions per image (min, avg, max): 21, 41, 81
Words per image (min, avg, max): 132, 290, 590
Nouns per image (min, avg, max): 37, 101, 197
Verbs per image (min, avg, max): 21, 45, 87

Vocab (unique words) per image (min, avg, max): 19, 62, 129
Nouns Vocab (unique words) per image (min, avg, max): 7, 25, 58
Verbs Vocab (unique words) per image (min, avg, max): 3, 15, 40
</pre>


<h3 align='center'>MSR-VTT</h3>

<pre>
---------- Training ----------
# Clips: 6513
# Captions: 130260
# Words: 1206706
# Nouns 433877 (35% of words)
# Verbs 165707 (13% of words)
Vocab: 23073
Nouns Vocab 15816 (68% of Vocab)
Verbs Vocab 7248 (31% of Vocab)

Captions per image (min, avg, max): 20, 20, 20
Words per image (min, avg, max): 92, 185, 419
Nouns per image (min, avg, max): 26, 66, 134
Verbs per image (min, avg, max): 3, 25, 62

Vocab (unique words) per image (min, avg, max): 15, 69, 154
Nouns Vocab (unique words) per image (min, avg, max): 7, 29, 76
Verbs Vocab (unique words) per image (min, avg, max): 1, 15, 42
</pre>
<pre>
---------- Validation ----------
# Clips: 497
# Captions: 9940
# Words: 91617
# Nouns 32989 (36% of words)
# Verbs 12623 (13% of words)
Vocab: 5936
Nouns Vocab 3795 (63% of Vocab)
Verbs Vocab 1771 (29% of Vocab)

Captions per image (min, avg, max): 20, 20, 20
Words per image (min, avg, max): 97, 184, 346
Nouns per image (min, avg, max): 36, 66, 124
Verbs per image (min, avg, max): 7, 25, 49

Vocab (unique words) per image (min, avg, max): 28, 69, 148
Nouns Vocab (unique words) per image (min, avg, max): 8, 29, 64
Verbs Vocab (unique words) per image (min, avg, max): 2, 15, 36
</pre>
<pre>
---------- Testing ----------
# Clips: 2990
# Captions: 59800
# Words: 556637
# Nouns 199267 (35% of words)
# Verbs 76971 (13% of words)
Vocab: 15680
Nouns Vocab 10593 (67% of Vocab)
Verbs Vocab 4879 (31% of Vocab)

Captions per image (min, avg, max): 20, 20, 20
Words per image (min, avg, max): 102, 186, 353
Nouns per image (min, avg, max): 34, 66, 122
Verbs per image (min, avg, max): 8, 25, 59

Vocab (unique words) per image (min, avg, max): 13, 70, 154
Nouns Vocab (unique words) per image (min, avg, max): 6, 29, 67
Verbs Vocab (unique words) per image (min, avg, max): 1, 15, 40
</pre>