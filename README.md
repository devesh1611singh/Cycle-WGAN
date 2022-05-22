# CycleGANs Improvements

In this study, I propose two image-to-image (I2I) translation models. The less studied, Cycle-WGAN model and the novel, Cycle-WGAN-SM model. 

## Cycle-WGAN
Since the adversarial loss in CycleGANs uses a min-max setup, CycleGANs themselves often face the GAN optimization setbacks. Mode collapse - a scenario where the generator model learns to create only a single or a small set of outputs, is common for CycleGAN models as well. If, for the generator $G(\cdot)$ doing a $A-to-B$ image translation and the discriminator (or critic) $D_B(\cdot)$ distinguishing the real images from the fakes in target data distribution $B$, the Wasserstein loss soft constrained by the gradient penatly is defined as:
   
    
$$
\begin{aligned} 
L_{WGANGP}(G, D_B, A, B) =   E_{b\sim{p_{data}}(b)}[D_B(b)]  -  E_{a\sim{p_{data}}(a)}[D_B(G(a))]  + \\ 
                             \lambda_{gp} \cdot E_{c\sim{p_{in.dist.}}(c)}[(\|\nabla_c D_B(c)\|_2 - 1)^2] 
\end{aligned}
$$
  
  
In this equation $in.dist.$ represents a random linearly interpolated distribution between the source data distribution $A$ and the target data distribution $B$. 

Then, Cycle-WGAN models replace the min-max adversarial setup of the CycleGAN's generator and discriminator networks with WGANs. The total loss for Cycle-WGAN models becomes:

$$
\begin{aligned} 
L(G, F, D_A, D_B)  =  L_{WGANGP}(G, D_B, A, B)   + \\ 
                     L_{WGANGP}(F, D_A, A, B)   + \\ 
                     \lambda_{c} \cdot L_{cyc}(G,F)   + \\ 
                     \lambda_{i} \cdot L_{identity}(G, F)   
\end{aligned}
$$

Where  $L_{cyc}(G,F)$ is the cycle consistency loss and the $L_{identity}(G, F)$ is the identity loss used in the standard CycleGANs.

See model architecture at:
![alt text](https://github.com/devesh1611singh/Cycle-WGAN/blob/main/ModelArchitecture/Cycle-WGAN.pdf?raw=true)

Notice the addition of $n_c$ steps - the number of steps a discriminator (or critic) trains for every generator step. Also notice that now the discriminator networks do not make a binary classification of - real or fake anymore, but rather output a distance metric $\in \mathbb{R}$.

A successful attempt at unsupervised I2I translation between the two domains - Aerial images and Maps, by the Cycle-WGAN could be seen in directory named Results.


## Cycle-WGAN-SM

Even with the Wasserstein metric regularization, the Cycle-WGAN model itroduce semantic errors in its translations. If an image is made of _content_ - shape, size, number and location of the entities in the image and _appearance_ (or _style_) - colour and texture of those entities. Then an optimal image-to-image translation model would only alter the _appearance_ of the entities in the image but not change the _content_ of the image. Changing the _content_ of the images during image translation fundamentally changes the meaning or semantics of the image. This could be a critical requirement of the I2I models, for example in healthcare setups.  

To control the semantic errors, a novel semantic loss (SM) was introduced to the Cycle-WGAN models. To evaluate the semantic loss, images before and after translation are passed through a feature extractor network $F_{ext}(\cdot)$, pre-trained on the ImageNet data set. Features from this network are collected just before the fully connected layers which are used for classification and are compared to calculate the loss. ResNet50 and MobileNetv2 were chosen as possible feature extractor networks.

Formally, semantic loss could be described as:

$$
\begin{aligned} 
L_{SM}(G)   =  E_{a\sim{p_{data}}(a)}[\|F_{ext}(G(a))-F_{ext}(a)\|_1]
\end{aligned}
$$

Semantic loss $L_{SM}(\cdot)$ (or simply SM), could be calculated individually for each generator. This allows for the possibility of applying it only for either of the two generator networks or both, depending on the scenario in question. Adding SM loss regularisation to both of the generator networks gives the total loss as:

$$
\begin{aligned} 
L(G, F, D_A, D_B)  =  L_{WGANGP}(G, D_B, A, B)   + \\ 
                     L_{WGANGP}(F, D_A, A, B)   + \\ 
                     \lambda_{c} \cdot L_{cyc}(G,F)   + \\ 
                     \lambda_{i} \cdot L_{identity}(G, F) + \\
                     \lambda_{sm} \cdot L_{SM}(G)  + \\
                     \lambda_{sm} \cdot L_{SM}(F)
\end{aligned}
$$

See model architecture at:
![alt text](https://github.com/devesh1611singh/Cycle-WGAN/blob/main/ModelArchitecture/Cycle-WGAN-SM.pdf?raw=true)


### References
1) The model architecture illustrations are inspired by the blog entry https://hardikbansal.github.io/CycleGANBlog/
2) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks (https://arxiv.org/pdf/1703.10593.pdf)
3) Improved Training of Wasserstein GANs (https://arxiv.org/pdf/1703.10593.pdf)
4) Imagenet: A large-scale hierarchical image database (https://image-net.org/static_files/papers/imagenet_cvpr09.pdf) 
5) Deep Residual Learning for Image Recognition (https://arxiv.org/pdf/1512.03385.pdf)
6) MobileNetV2: Inverted Residuals and Linear Bottlenecks (https://arxiv.org/pdf/1801.04381.pdf)
