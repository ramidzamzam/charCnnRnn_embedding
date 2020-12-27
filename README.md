# CharCnnRnn Text Embedding 
A data processing pipeline (main script data_prep.py) that takes screenshots, and the images 
(screenshots) description text files to generate CharCnnRnn text embedding tensors 
using pre-trained models, ConvAutoencoder for image feature extraction and CharCnnRnn
used to create the text embedding. Both models included and can be trained on a custom dataset.<br/>
The generated output text embedding tensors' files can be used as input to [stackGAN](https://github.com/hanzhanggit/StackGAN). 

![alt text](flow.png)