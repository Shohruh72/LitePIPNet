# LitePIPNet Facial Landmark Detection

### This repository contains the implementation of PIPNet, a robust approach for facial landmark detection using a deep learning model based on ResNet architectures.
![Vizualization](https://github.com/Shohruh72/PIPNet/blob/main/outputs/result.gif)

[Click here to watch the video](https://www.youtube.com/watch?v=cxi1WQr-HKE)


### Key Achievements
#### Exceptional Model Performance on the 300W Dataset

## Features
* #### Utilizes ResNet as the backbone for the PIPNet model.
* #### Supports training, testing, and real-time demo modes.
* #### Includes a 300W dataset loader and loss computation.
* #### Implements a face detector for real-time landmark detection in videos.
* #### Designed for easy customization and scalability to accommodate research and development needs.
          
### Requirements
```bash
conda create -n PyTorch python=3.8
conda activate PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```           
## Usage
**Datasets: [300W](https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/)**
* Download the datasets from official sources.
**The folder structure should look like this:**
 * #### afw
 * #### helen
 * #### ibug
 * #### lfpw  

 **Run the below command for dataset preparation**
```bash
$ python -c 'from utils.util import DataGenerator; gen = DataGenerator("../Datasets_path/"); gen.run()'
```

### Training
_**To train the model, run:**_
* Configure your dataset path in main.py for training

```bash
$ python main.py --train
```
### Testing
_**For testing the model, use:**_
* Configure your dataset path in main.py for testing

```bash
$ python main.py --test
```

### Real-Time Demo
**_To run the real-time facial landmark detection:_**
```bash
$ python main.py --demo


### Results
| Backbone   | Epochs | Test NME |                                                                 Pretrained weights |
|:----------:|:------:|---------:|-----------------------------------------------------------------------------------:|
| MobileOne  |   soon  |     soon |  [model]() |
| MobileOne  |   soon  |     soon |  [model]() |
| MobileOne  |   soon  |     soon |  [model]() |

##### Reference
* https://github.com/jhb86253817/PIPNet
