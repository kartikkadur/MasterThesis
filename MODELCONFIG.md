# Model Configurations
This page lists the model components and different configurations that was evaluated in this work.

## Model Components
### Content Encoder
* Encodes image content that are domain invarient into a content feature space
* The encoder model is constructed using several downsampling blocks followed by several residual blocks

### Style Encoder
* Encodes domain specific style features into a style feature space.
* The style feature space is a vector of predefined shape encoding different information representing the style information.
* The model is constructed using several downsampling blocks followed by a Adaptive Average Pooling layer at the end to reduce the spatial dimention to a vector.

### Decoder
* Utilizes content and style encoded features to generate translated image.
* Decoder model consists of several residual blocks followed by upsampling blocks to being the features back to the input image shape.

### Discriminators
* Domain discriminator classifies an image as real/fake and performs domain classification by predicting the correct target domain.
* Content discriminator classifies the encoded content features into different domain classes.

## Configurations
### Base Model
Injects style using depth-wise concatination operation on the encoded content features.
![base_model](https://github.com/kartikkadur/MasterThesis/blob/main/images/basemodel.png)
### AdaINModel
Injects style using Adaptive Instance Normalization (AdaIN) operations in the decoder residual blocks.
![adain_model](https://github.com/kartikkadur/MasterThesis/blob/main/images/adainmodel.png)

## Synthesis techniques
### Reference-based Synthesis
![ref](https://github.com/kartikkadur/MasterThesis/blob/main/images/reference_based_synthesis.png)
### Random-based Synthesis
![ran](https://github.com/kartikkadur/MasterThesis/blob/main/images/random_based_synthesis.png)
