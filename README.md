## Fast Style Transfer in [TensorFlow](https://github.com/tensorflow/tensorflow)

Add styles from famous paintings to any photo in a fraction of a second! [You can even style videos!](#video-stylization)

<p align = 'center'>
<img src = 'examples/style/udnie.jpg' height = '246px'>
<img src = 'examples/content/stata.jpg' height = '246px'>
<a href = 'examples/results/stata_udnie.jpg'><img src = 'examples/results/stata_udnie_header.jpg' width = '627px'></a>
</p>
<p align = 'center'>
It takes 100ms on a 2015 Titan X to style the MIT Stata Center (1024×680) like Udnie, by Francis Picabia.
</p>

Our implementation is based off of a combination of Gatys' [A Neural Algorithm of Artistic Style](https://arxiv.org/abs/1508.06576), Johnson's [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](http://cs.stanford.edu/people/jcjohns/eccv16/), and Ulyanov's [Instance Normalization](https://arxiv.org/abs/1607.08022). 

#### Don't have access to a GPU but want to train a style or transform large batches of images?
Contact me at engstrom@mit.edu for rates.

## Video Stylization 
Here we transformed every frame in a video, then combined the results. [Click to go to the full demo on YouTube!](https://www.youtube.com/watch?v=xVJwwWQlQ1o) The style here is Udnie, as above.
<div align = 'center'>
     <a href = 'https://www.youtube.com/watch?v=xVJwwWQlQ1o'>
        <img src = 'examples/results/fox_udnie.gif' alt = 'Stylized fox video. Click to go to YouTube!' width = '800px' height = '400px'>
     </a>
</div>

See how to generate these videos [here](#stylizing-video)!

## Image Stylization
We added styles from various paintings to a photo of Chicago. Click on thumbnails to see full applied style images.
<div align='center'>
<img src = 'examples/content/chicago.jpg' height="200px">
</div>
     
<div align = 'center'>
<a href = 'examples/style/wave.jpg'><img src = 'examples/thumbs/wave.jpg' height = '200px'></a>
<img src = 'examples/results/chicago_wave.jpg' height = '200px'>
<img src = 'examples/results/chicago_udnie.jpg' height = '200px'>
<a href = 'examples/style/udnie.jpg'><img src = 'examples/thumbs/udnie.jpg' height = '200px'></a>
<br>
<a href = 'examples/style/rain_princess.jpg'><img src = 'examples/thumbs/rain_princess.jpg' height = '200px'></a>
<img src = 'examples/results/chicago_rain_princess.jpg' height = '200px'>
<img src = 'examples/results/chicago_la_muse.jpg' height = '200px'>
<a href = 'examples/style/la_muse.jpg'><img src = 'examples/thumbs/la_muse.jpg' height = '200px'></a>

<br>
<a href = 'examples/style/the_shipwreck_of_the_minotaur.jpg'><img src = 'examples/thumbs/the_shipwreck_of_the_minotaur.jpg' height = '200px'></a>
<img src = 'examples/results/chicago_wreck.jpg' height = '200px'>
<img src = 'examples/results/chicago_the_scream.jpg' height = '200px'>
<a href = 'examples/style/the_scream.jpg'><img src = 'examples/thumbs/the_scream.jpg' height = '200px'></a>
</div>

## Implementation Details
Our implementation uses TensorFlow to train a fast style transfer network. We use roughly the same transformation network as described in Johnson, except that batch normalization is replaced with Ulyanov's instance normalization, and the scaling/offset of the output `tanh` layer is slightly different. We use a loss function close to the one described in Gatys, using VGG19 instead of VGG16 and typically using "shallower" layers than in Johnson's implementation (e.g. we use `relu1_1` rather than `relu1_2`). Empirically, this results in larger scale style features in transformations.

## Documentation
### Training Style Transfer Networks
Use `style.py` to train a new style transfer network. Run `python style.py` to view all the possible parameters. Training takes 4-6 hours on a Maxwell Titan X. [More detailed documentation here](docs.md#style). **Before you run this, you should run `setup.sh`**. Example usage:

    python style.py --style path/to/style/img.jpg \
      --checkpoint-dir checkpoint/path \
      --test path/to/test/img.jpg \
      --test-dir path/to/test/dir \
      --content-weight 1.5e1 \
      --checkpoint-iterations 1000 \
      --batch-size 20

### Evaluating Style Transfer Networks
Use `evaluate.py` to evaluate a style transfer network. Run `python evaluate.py` to view all the possible parameters. Evaluation takes 100 ms per frame (when batch size is 1) on a Maxwell Titan X. [More detailed documentation here](docs.md#evaluate). Takes several seconds per frame on a CPU. **Models for evaluation are [located here](https://drive.google.com/drive/folders/0B9jhaT37ydSyRk9UX0wwX3BpMzQ?usp=sharing)**. Example usage:

    python evaluate.py --checkpoint path/to/style/model.ckpt \
      --in-path dir/of/test/imgs/ \
      --out-path dir/for/results/

### Stylizing Video
Use `transform_video.py` to transfer style into a video. Run `python transform_video.py` to view all the possible parameters. Requires `ffmpeg`. [More detailed documentation here](docs.md#video). Example usage:

    python transform_video.py --in-path path/to/input/vid.mp4 \
      --checkpoint path/to/style/model.ckpt \
      --out-path out/video.mp4 \
      --device /gpu:0 \
      --batch-size 4

### Requirements
You will need the following to run the above:
- TensorFlow 0.11.0
- Python 2.7.9, Pillow 3.4.2, scipy 0.18.1, numpy 1.11.2
- If you want to train (and don't want to wait for 4 months):
  - A decent GPU
  - All the required NVIDIA software to run TF on a GPU (cuda, etc)
- ffmpeg 3.1.3 if you want to stylize video

### Citation
```
  @misc{engstrom2016faststyletransfer,
    author = {Logan Engstrom},
    title = {Fast Style Transfer},
    year = {2016},
    howpublished = {\url{https://github.com/lengstrom/fast-style-transfer/}},
    note = {commit xxxxxxx}
  }
```

### Attributions/Thanks
- This project could not have happened without the advice (and GPU access) given by [Anish Athalye](http://www.anishathalye.com/). 
  - The project also borrowed some code from Anish's [Neural Style](https://github.com/anishathalye/neural-style/)
- Some readme/docs formatting was borrowed from Justin Johnson's [Fast Neural Style](https://github.com/jcjohnson/fast-neural-style)
- The image of the Stata Center at the very beginning of the README was taken by [Juan Paulo](https://juanpaulo.me/)

### License
Copyright (c) 2016 Logan Engstrom. Contact me for commercial use (email: engstrom at my university's domain dot edu). Free for research/noncommercial use, as long as proper attribution is given and this copyright notice is retained.

___

## style.py 

`style.py` trains networks that can transfer styles from artwork into images.

**Flags**
- `--checkpoint-dir`: Directory to save checkpoint in. Required.
- `--style`: Path to style image. Required.
- `--train-path`: Path to training images folder. Default: `data/train2014`.
- `--test`: Path to content image to test network on at at every checkpoint iteration. Default: no image.
- `--test-dir`: Path to directory to save test images in. Required if `--test` is passed a value.
- `--epochs`: Epochs to train for. Default: `2`.
- `--batch_size`: Batch size for training. Default: `4`.
- `--checkpoint-iterations`: Number of iterations to go for between checkpoints. Default: `2000`.
- `--vgg-path`: Path to VGG19 network (default). Can pass VGG16 if you want to try out other loss functions. Default: `data/imagenet-vgg-verydeep-19.mat`.
- `--content-weight`: Weight of content in loss function. Default: `7.5e0`.
- `--style-weight`: Weight of style in loss function. Default: `1e2`.
- `--tv-weight`: Weight of total variation term in loss function. Default: `2e2`.
- `--learning-rate`: Learning rate for optimizer. Default: `1e-3`.
- `--slow`: For debugging loss function. Direct optimization on pixels using Gatys' approach. Uses `test` image as content value, `test_dir` for saving fully optimized images.


## evaluate.py
`evaluate.py` evaluates trained networks given a checkpoint directory. If evaluating images from a directory, every image in the directory must have the same dimensions.

**Flags**
- `--checkpoint`: Directory or `ckpt` file to load checkpoint from. Required.
- `--in-path`: Path of image or directory of images to transform. Required.
- `--out-path`: Out path of transformed image or out directory to put transformed images from in directory (if `in_path` is a directory). Required.
- `--device`: Device used to transform image. Default: `/cpu:0`.
- `--batch-size`: Batch size used to evaluate images. In particular meant for directory transformations. Default: `4`.
- `--allow-different-dimensions`: Allow different image dimensions. Default: not enabled

## transform_video.py
`transform_video.py` transforms videos into stylized videos given a style transfer net.

**Flags**
- `--checkpoint-dir`: Directory or `ckpt` file to load checkpoint from. Required.
- `--in-path`: Path to video to transfer style to. Required.
- `--out-path`: Path to out video. Required.
- `--tmp-dir`: Directory to put temporary processing files in. Will generate a dir if you do not pass it a path. Will delete tmpdir afterwards. Default: randomly generates invisible dir, then deletes it after execution completion.
- `--device`: Device to evaluate frames with. Default: `/gpu:0`.
- `--batch-size`: Batch size for evaluating images. Default: `4`.
