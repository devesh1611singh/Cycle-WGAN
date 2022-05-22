'''
Author: Devesh Singh
Date: 22.05.2022

Explains the model Cycle-WGAN. Where inside the CycleGAN architecture, the generator and discriminator models are trained used Wasserstein distance metric soft 
constrained by the gradient penalty.  
'''

from numpy import True_
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix
import time
from datetime import datetime
import Manage as m

m.limit_GPU(2)

##--------------------------------------------  INPUT PIPELINE  --------------------------------------------
n_critic = 5 
batch_size = 1 
buffer_size = 1000
autotune = tf.data.experimental.AUTOTUNE

real_train_dataset = tf.data.Dataset.from_tensor_slices((m.files_names_real(path=m.data_real_train, RemoveBags=False))).shuffle(buffer_size)
real_train_dataset = real_train_dataset.map(m.parse_function).batch(batch_size)

sim_dataset = tf.data.Dataset.from_tensor_slices((m.files_names_sim(path=m.data_sim,RemoveEmptyFiles= True, RemoveBags=False))).shuffle(buffer_size)
sim_dataset = sim_dataset.map(m.parse_function_sim).batch(batch_size)

sample_real = next(iter(real_train_dataset))
sample_sim = next(iter(sim_dataset))  


##--------------------------------------------  Import and reuse the Pix2Pix models  --------------------------------------------

OUTPUT_CHANNELS = 3            
INPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(output_channels=OUTPUT_CHANNELS, norm_type='instancenorm', INPUT_CHANNELS=INPUT_CHANNELS, ch=1)
generator_f = pix2pix.unet_generator(output_channels=OUTPUT_CHANNELS, norm_type='instancenorm', INPUT_CHANNELS=INPUT_CHANNELS, ch=1)

INPUT_CHANNELS = 3
discriminator_x = pix2pix.discriminator(norm_type='instancenorm', INPUT_CHANNELS=INPUT_CHANNELS, target=False, ch=1)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', INPUT_CHANNELS=INPUT_CHANNELS, target=False, ch=1)


##--------------------------------------------  LOSS FUNCTIONS  --------------------------------------------
LAMBDA = 10
GP_LAMBDA=10

def discriminator_loss_WASS_alt(real, generated):
  return tf.reduce_mean(real) - tf.reduce_mean(generated)

def generator_loss_WASS_alt(generated):
  return tf.reduce_mean(generated)

def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1

def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss

@tf.function
def gradient_panelty(Discriminator,real,generated):
    alpha = tf.random.uniform(
        shape=[batch_size,1], 
        minval=0.,
        maxval=1.
    )

    #Line6
    interpolates = alpha*real + ((1-alpha)*generated)
  
    disc_interpolates = Discriminator(interpolates)
    gradients = tf.gradients(disc_interpolates, [interpolates])
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients[0]), axis=[1]))      #abs(gradients)
    gradient_penalty = tf.reduce_mean((slopes-1)**2)                        # ( abs(gradients) -1 )^2 
 
    return GP_LAMBDA*gradient_penalty

lr = 0.0001
b1 = 0.9
b2 = 0.999
generator_g_optimizer = tf.keras.optimizers.Adam(lr, b1, b2)     
generator_f_optimizer = tf.keras.optimizers.Adam(lr, b1, b2)       

#Training with a bigger learning rate for the discriminator models
eta = 5
discriminator_x_optimizer = tf.keras.optimizers.Adam(lr*eta, b1, b2)      
discriminator_y_optimizer = tf.keras.optimizers.Adam(lr*eta, b1, b2)  


##--------------------------------------------  CHECKPOINTS  --------------------------------------------


ckpt = tf.train.Checkpoint(generator_g=generator_g,
                            generator_f=generator_f,
                            discriminator_x=discriminator_x,
                            discriminator_y=discriminator_y,
                            generator_g_optimizer=generator_g_optimizer,
                            generator_f_optimizer=generator_f_optimizer,
                            discriminator_x_optimizer=discriminator_x_optimizer,
                            discriminator_y_optimizer=discriminator_y_optimizer)



# if a checkpoint exists, restore the latest checkpoint.
#if ckpt_manager.latest_checkpoint:
#   ckpt.restore(ckpt_manager.latest_checkpoint)
#   print ('Latest checkpoint restored!!')

##--------------------------------------------  TRAINING  --------------------------------------------

@tf.function
def train_step_GP(real_x, real_y):
#from the Algo.1 (p4) in https://proceedings.neurips.cc/paper/2017/file/892c3b1c6dccd52936e27cbd0ff683d6-Paper.pdf
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X
    # X: real image domain
    # Y: simulated image domain

    #Line 2  + in line 3 m=1, so that loop collapses
    for i in range(n_critic):

      #Line 4: New images of each iteration of n_critic/inner loop
      if i > 0:
        real_x = m.encode(next(iter(real_train_dataset)))
        real_y = m.encode(next(iter(sim_dataset)),True)
      #Line 5
      fake_y = generator_g(real_x, training=True)     
      fake_x = generator_f(real_y, training=True)

      disc_real_x = discriminator_x(real_x, training=True)
      disc_real_y = discriminator_y(real_y, training=True)
      disc_fake_x = discriminator_x(fake_x, training=True)
      disc_fake_y = discriminator_y(fake_y, training=True)


      disc_x_normal = discriminator_loss_WASS_alt(disc_real_x, disc_fake_x)
      disc_x_wassgp = gradient_panelty(discriminator_x, real_x, fake_x)
      disc_x_loss =  disc_x_normal + disc_x_wassgp
      
      disc_y_normal = discriminator_loss_WASS_alt(disc_real_y, disc_fake_y)
      disc_y_wassgp = gradient_panelty(discriminator_y, real_y, fake_y)     
      disc_y_loss = disc_y_normal + disc_y_wassgp

      #Line 9 
      discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
      discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)
      discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
      discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    cycled_x = generator_f(fake_y, training=True)
    cycled_y = generator_g(fake_x, training=True)
    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    # calculate the loss
    #Claculating again in the outer loop. With D(.) trained for n_critic steps
    # Line 11 
    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)
    gen_g_loss = generator_loss_WASS_alt(disc_fake_y)
    gen_f_loss = generator_loss_WASS_alt(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)
    
    # Identity loss
    identity_loss_real_y_same_y = identity_loss(real_y, same_y)
    identity_loss_real_x_same_x = identity_loss(real_x, same_x)

    # Total generator loss = adversarial loss + cycle loss + identity loss 
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss_real_y_same_y 
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss_real_x_same_x 

    #Line 12  
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

  return [gen_g_loss, total_cycle_loss, identity_loss_real_y_same_y, 0.0, total_gen_g_loss,
          gen_f_loss, total_cycle_loss, identity_loss_real_x_same_x, 0.0, total_gen_f_loss,
          disc_x_normal, disc_x_wassgp, disc_x_loss,
          disc_y_normal , disc_y_wassgp, disc_y_loss]         #disc_loss from the last iteration of inner loop (n_critic)




##--------------------------------------------  Main Loop  --------------------------------------------
def main(EPOCHS = 10, testing=False , experiment_flag='_testing'):
  m.set_channels([0,1,2])    
  
  time_path = datetime.now().strftime("%Y%m%d-%I%M%S")

  log_dir="logs/"
  summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + time_path+experiment_flag)

  epoch_loss = [0,0,0,0,0,
                0,0,0,0,0,
                0,0,0,
                0,0,0]

  checkpoint_path = "./checkpoints/train/" + time_path+experiment_flag
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)  

  for epoch in range(EPOCHS):
    start = time.time()
    n = 1
    for image_x, image_y in zip(real_train_dataset, sim_dataset):
      step_loss = train_step_GP(m.encode(image_x) ,m.encode(image_y, True) ) 

      for i in range(len(step_loss)):
        epoch_loss[i] = ( epoch_loss[i]*(n-1) + step_loss[i] ) / n

      if n % 200 == 0: 
        print ('.', end='')
      
      if testing:
        if n==5:
          break

      n+= 1
    end=time.time()
    
    with summary_writer.as_default():
      for i in range(len(epoch_loss)):
        tf.summary.scalar(m.cols[i], epoch_loss[i], step=epoch)

    m.generate_images(generator_f, sample_sim,time_path+experiment_flag,epoch)

    if (epoch + 1) % 5 == 0:
      ckpt_save_path = ckpt_manager.save()
      print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, end-start))

  print('--Trained--')
  m.trained_check(generator_f,time_path+experiment_flag)


main(20,False,'_Cycle-WGAN')
