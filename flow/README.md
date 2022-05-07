Flow based models have attracted a lot of attention due to its ability to predict exact loglikelihood and tracts the exact latent variable inference. 

In GANs, datapoints can usually not be directly represented in a latent space, as they have no encoder and might not have full support over the data distribution.
This is not the case for reversible generative models and VAEs, which allow for various applications such as interpolations between datapoints and meaningful modifications of existing datapoints.
We would be using one such interpolation for latent attributes manipulations, leading to meaningful manipulations to the existing features. 

We would be drawing meaningful results from the GLOW based model which is an advancement to the initial flow based model, due to the presence of certain layers such as affine coupling layer and the invertible 1x1 convolutional layer. 

The affine coupling layer is an efficient reversible transformation for reversing the function and calculating the log likelihood determinant with lesser compute intensity. 
Each step of flow above should be preceded by some kind of permutation of the variables that ensures that after sufficient steps of flow, each dimensions can affect every other
dimension. Our invertible 1x1 convolution is a generalization of such permutations. In experiments we compare these three choices.

One of the changes we tried to the existing model is the way we decompose the 40 latent attributes (bangs, age, gender etc.), the paper proposed to consider the presence of the latent attribute as a positive vector and its absence to be negative and then summing it up. I have hereby tried to take the average of some sample of pictures for the calculation of the latent attribute vectors. 

I have added some of the results.  
