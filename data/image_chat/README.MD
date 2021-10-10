# Image Captioning Model
In this repo we used the from [https://github.com/ruotianluo/ImageCaptioning.pytorch](https://github.com/ruotianluo/ImageCaptioning.pytorch). We downloaded the **vilbert-12-in-1** model from the [https://github.com/ruotianluo/ImageCaptioning.pytorch/blob/master/MODEL_ZOO.md](MODEL_ZOO) and we generate the image caption by 

```
python tools/eval.py --model model.pth --infos_path infos.pkl --image_folder blah --num_images 10
```

In ```results_caption.json``` you can find the raw results for the validation set and test set, while in the ```img_valid.json``` and ```img_test.json``` the caption aligned with the dialogues. 