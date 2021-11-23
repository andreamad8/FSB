# Image Captioning Model
In this repo we used the from [https://github.com/ruotianluo/ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch). We downloaded the **vilbert-12-in-1** model from the [https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/MODEL_ZOO.md](MODEL_ZOO) and we generate the image caption by 

```
python tools/eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

In ```results_caption.json``` you can find the raw results for the validation set and test set, while in the ```img_valid.json``` and ```img_test.json``` the caption aligned with the dialogues. 


# How to install the model
```
pip install gdown
git clone https://github.com/ruotianluo/ImageCaptioning.pytorch.git
cd ImageCaptioning.pytorch
python -m pip install -e .
pip install lmdbdict
pip install scikit-image
pip install git+https://github.com/ruotianluo/meshed-memory-transformer.git
gdown https://drive.google.com/uc\?id\=1VmUzgu0qlmCMqM1ajoOZxOXP3hiC_qlL
gdown https://drive.google.com/uc\?id\=1zQe00W02veVYq-hdq5WsPOS3OPkNdq79 
cd data
mkdir imagenet_weights
cd imagenet_weights
gdown https://drive.google.com/uc\?id\=0B7fNdx_jAqhtSmdCNDVOVVdINWs
gdown https://drive.google.com/uc\?id\=0B7fNdx_jAqhtckNGQ2FLd25fa3c
cd ../..
```