<h1 align='center'>Feature Generation</h1>
<p align="center">Details on how to perform feature extraction. You can extract video features with (<a href="https://arxiv.org/pdf/1705.07750.pdf">I3D</a>, <a href="https://arxiv.org/abs/1812.03982">SlowFast</a>), image features (TODO) and object detections with features (YOLOv3). For all below we extract features on keyframes, which in our case is every <b>1 second</b> - so if a video is 15 fps, keyframes are every 15 frames, while a 60 fps video has keyframes at every 60 frames. This is done as real life time is consistent with motion unlike the framerates at which that motion is captured, permitting better temporal modelling.</p>

<h2 align='center'></h2>
<h2 align='center'>Video Features</h2>

<p align="center">For <a href="https://arxiv.org/pdf/1705.07750.pdf">I3D</a> video features we use windows of size <b>32 frames</b> with <b>stride 1</b>, and for <a href="https://arxiv.org/abs/1812.03982">SlowFast</a> we use <b>stride 2</b>, both are <b>centred</b> around the keyframe. For <a href="https://arxiv.org/pdf/1705.07750.pdf">I3D</a> we extract final dense prior to classification (size 2048) layer activations while for <a href="https://arxiv.org/abs/1812.03982">SlowFast</a> we similarly extract the final dense layer prior to classification (size 2304). Both the <a href="https://arxiv.org/pdf/1705.07750.pdf">I3D</a> and <a href="https://arxiv.org/abs/1812.03982">SlowFast</a> networks are pretrained on the <a href="https://deepmind.com/research/open-source/kinetics">Kinetics 400</a> dataset.</p>

<p align="center">To extract <a href="https://arxiv.org/pdf/1705.07750.pdf">I3D</a> features from the <a href="https://vsubhashini.github.io/s2vt.html">MSVD</a> dataset you can use:</p>

<pre>
path/to/VidCap$ python generate_features.py --cfg features/runs/MSVD_i3d.yaml
</pre>

<p align="center">Other options are (for <a href="https://www.microsoft.com/en-us/research/publication/msr-vtt-a-large-video-description-dataset-for-bridging-video-and-language/">MSR-VTT</a> dataset and the <a href="https://arxiv.org/abs/1812.03982">SlowFast</a> network):</p>
<pre>
path/to/VidCap$ python generate_features.py --cfg features/runs/MSVD_slowfast.yaml
path/to/VidCap$ python generate_features.py --cfg features/runs/MSRVTT_i3d.yaml
path/to/VidCap$ python generate_features.py --cfg features/runs/MSRVTT_slowfast.yaml
</pre>

<h2 align='center'></h2>
<h2 align='center'>Image (Framewise) Features</h2>
<h3 align='center'>todo</h3>
<p align="center"></p>

<h2 align='center'></h2>
<h2 align='center'>Object Features</h2>
<h3 align='center'>todo</h3>
<p align="center"></p>