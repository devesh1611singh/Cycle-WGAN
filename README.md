# CyccleGANs Improvements

In this study I propose two image-to-image transaltion models. The less studied, Cycle-WGAN model and the novel, Cycle-WGAN-SM model. 

## Cycle-WGAN
Since the adversarial loss in CycleGANs uses a min-max setup, CycleGANs themselves often face the GAN optimization setbacks. Mode collapse - a scenario where the generator model learns to create only a single or a small set of outputs, is common for CycleGAN models as well. If, for the generator $G(.)$ doing a $A-to-B$ image translation and the discriminator (or critic) $D_B(.)$ distinguishuing the real images from the fakes in target data distribution $B$, the Wasserstein loss soft constrained by the gradient penatly is defained as:
   
    
$$
\begin{aligned} 
L_{WGANGP}(G, D_B, A, B) =   E_{b\sim{p_{data}}(b)}[D_B(b)]  -  E_{a\sim{p_{data}}(a)}[D_B(G(a))]  + \\ 
                             \lambda_{gp}E_{c\sim{p_{in.dist.}}(c)}[(\|\nabla_c D_B(c)\|_2 - 1)^2] 
\end{aligned}
$$
  
  
In this equation $in.dist.$ represents a random linearly interpolated distribution between the source data distribution $A$ and the target data distribution $B$. 

Then, Cycle-WGAN models reaplace the min-max adversarial setup of the CycleGAN's generator and discriminator netwroks with WGANs. The total loss for Cycle-WGAN models become:

$$
\begin{aligned} 
L(G, F, D_A, D_B)  =  L_{WGANGP}(G, D_B, A, B)   + \\ 
                     L_{WGANGP}(F, D_A, A, B)   + \\ 
                     \lambda_{c}L_{cyc}(G,F)   + \\ 
                     \lambda_{i}L_{identity}(G, F)   
\end{aligned}
$$

Where  $\lambda_{c}L_{cyc}(G,F)$ is the cycle consistency loss and the $\lambda_{i}L_{identity}(G, F)$ is the identity loss used in the standard CycleGANs.

![alt text](https://github.com/[devesh1611singh]/[Cycle-WGAN]/tree/[main]/[ModelArchitecture]/Cycle-WGAN.pdf?raw=true)
