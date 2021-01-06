# CharCnnRnn Text Embedding 
A data processing pipeline (main script data_prep.py) that takes screenshots, and the images 
(screenshots) description text files to generate CharCnnRnn text embedding tensors 
using pre-trained models, ConvAutoencoder for image feature extraction and CharCnnRnn
used to create the text embedding. Both models included and can be trained on a custom dataset.<br/>
The generated output text embedding tensors' files can be used as input to [stackGAN](https://github.com/hanzhanggit/StackGAN). 

![alt text](flow.png)

## Train Custom Dataset
1. Train the conv_autoencoder model. <br>
   run ```python train.py .../dataset .../output 0.001 40``` <br>
   Inside /dataset should be two folders, test and train each contains 64x64 or 256x256 gray .png images.
2. Train char_embedding model. <br>
   run ``` python train.py .../images_data.json .../output 0.001 20 fixed_gru cvpr img_64x64_path```<br>
   The input is path to json file inside a folder contains the images folder 64x64_images or 256x256_images, see below example. <br>
   ```[{
    "text": "Hello Woeld! ...\n",
    "img_64x64_path": "/64x64_images/64x64_1609088299704.png",
    "img_256x256_path": "/256x256_images/256x256_1609088299704.png"
  }]```

