# Multi-modal and Multi-domain Image-to-Image translation
This work aims at developing deep learning models to perform multi-modal and multi-domain image translations between different weather domains. The work propose three deep learning model configurations that combine several state-of-the art ideas in performing image translations. The ideas from GANs and VAEs are utilized in building these configurations to achieve high quality and diversity translations. See the file [MODELCONFIG.md](https://github.com/kartikkadur/MasterThesis/blob/main/MODELCONFIG.md) for the architecture of these configurations.
### Example Translations
![Translation](https://github.com/kartikkadur/MasterThesis/blob/main/images/translation.png)

## Dependencies
You can install all the dependencies by 
```
pip install requirements.txt
```
or manually install the dependencies listed below:
```
numpy >= 1.21.5
python >=3.7
pytorch >= 1.9 with cuda 11.3
torchvision >= 0.8.2
tensorboard >= 2.7.0
Pillow = 8.1.2 (doesn't support python 3.10)
pytorch-fid = 0.2.1 (if you want to compute FID values)
lpips = 0.1.4 (if you want to computer LPIPS scores)
```
## Getting Started


### Datasets
The training datasets containing weather images can be downloaded from the link: [Image2Weather](https://www.cs.ccu.edu.tw/~wtchu/projects/Weather/index.html)

### Usage
#### Generate sample images
* Directly run the script to translate images using `AdaINModel` configuration.
* Edit the `--model` option to choose between `BaseModel` or `AdaINModel` configurations to play around with different configurations.
```
bash ./sample.sh
```
#### Train the model
Run `python train.py -h` for more information on commandline options
```
bash ./train.sh
```

## Translations
![Process](https://github.com/kartikkadur/MasterThesis/blob/main/images/process.jpg)
