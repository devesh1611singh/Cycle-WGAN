'''
Author: Devesh Singh
Date: 22.05.2022 

A Util file, used to manage the preprocessing, loading and saving actions of the models. 
'''


import tensorflow as tf
import os
import matplotlib.pyplot as plt
import json
import pandas as pd
import seaborn as sns  
from tensorflow.keras.models import Model

##-------------------------------------------- global variables --------------------------------------------

data_sim = '/sim/'                      #Path to Simulated Dataset
data_real_train = '/train/'             #Path to train split of the Real Dataset
data_real_test = '/test/'               #Path to test split of the Real Dataset
data_real_val = '/val/'                 #Path to validation split of the Real Dataset
save_train = '/dir/'                    #Path to save the trained models and results
examples_file =  '/Examples.json'       #Path to json file, indexing to all the samples simulated images which are tested for all the experiments 

img_height_width = 256                  #Images are rendered to a fixed size of 256x255 pixels
kept_channels = [0,1,2,3,4,5,6,7]       #Channels to keep in the one hot encoding
omitted_channels = []
pretrained_model = None

cols = ['Generator G Wass', 'Cycle consitency loss: G', 'Identity loss: Y', 'Semantic Loss: G', 'All Components Generator G',
        'Generator F Wass', 'Cycle consitency loss: F', 'Identity loss: X', 'Semantic Loss: F', 'All Components Generator F',
        'Discriminator X Wass', 'Discriminator X_GP', 'All Components Discriminator X',
        'Discriminator Y Wass', 'Discriminator Y_GP', 'All Components Discriminator Y']


##-------------------------------------------- Graphical engagement with the results  --------------------------------------------

def save_fig(plt,time,epoch=0,flag='_'):
    path = os.path.join(save_train, time)
    path = os.path.join(path,'plots/')
    os.makedirs(path, exist_ok=True)

    file = os.path.join(path, str(epoch) + flag + '.png')  
    plt.savefig(file, dpi=300)


def print_data(ip,op, time_path, epoch, flag):
    '''
    prints images ip and op images
    ip: input to a generator, as [H,W,C=9]
    op: generator's output, as [H,W,C=9]
    '''

    ip_raw = ip[:,:,0]
    ip_segmask = ip[:,:,1:]

    op_raw = op[:,:,0]
    op_segmask = op[:,:, 1:]

    #Reversing one hot encoding
    ip_segmask = tf.argmax(ip_segmask, -1)
    op_segmask = tf.argmax(op_segmask, -1) 
    
    ip_segmask = tf.image.convert_image_dtype(ip_segmask, tf.float32)
    op_segmask = tf.image.convert_image_dtype(op_segmask, tf.float32)

    fig, axs = plt.subplots(2, 2, figsize=(8,8))

    axs[0, 0].imshow(ip_raw, cmap='gray')
    axs[0, 0].set_title('Input Image')
    axs[0, 1].imshow(op_raw, cmap='gray')
    axs[0, 1].set_title('Predicted Image')
    axs[1, 0].imshow(ip_segmask, cmap='gray')
    axs[1, 0].set_title('Input Seg Maks')
    axs[1, 1].imshow(op_segmask, cmap='gray')
    axs[1, 1].set_title('Predicted Seg Mask')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    save_fig(fig,time_path,epoch,flag )


##-------------------------------------------- pre-processing for tf models --------------------------------------------

def files_names_sim(path, RemoveEmptyFiles=False, RemoveBags=False):
    '''
    :param path: directory which contains both raw and segmentation label images
    :return: seprate lists of file names for
    '''
    files = os.listdir(path)
    filesnames_raw,filesnames_labels = [],[]
    filesnames_all = []


    if RemoveBags==False:
      with open ("/data/people/tsingde/thesis/bag_only_dict.json","r") as f_bag:
        bag_only_dict = json.load(f_bag)
      with open ("/data/people/tsingde/thesis/box_and_bag_dict.json","r") as f_box_bag:
        box_bag_dict = json.load(f_box_bag)
      filesnames_all += list(bag_only_dict.keys())
      filesnames_all += list(box_bag_dict.keys())
    

    if RemoveEmptyFiles==False:
      with open ("/data/people/tsingde/thesis/empty_dict.json","r") as f_empty:
        empty_dict = json.load(f_empty)
      filesnames_all += list(empty_dict.keys())


    with open ("/data/people/tsingde/thesis/box_only_dict.json","r") as f_box:
      box_only_dict = json.load(f_box)
    #Repeating box images to correct the ratio of labels in the simulated domain
    filesnames_all +=  list(box_only_dict.keys()) + list(box_only_dict.keys())
    

    filesnames_raw =    [path+ f + '_rgb.png'  for f in filesnames_all]
    filesnames_labels = [path+ f + '_seg_mask_rgb.png'  for f in filesnames_all]


    filesnames_raw.sort()
    filesnames_labels.sort()

    return filesnames_raw,filesnames_labels


def files_names_real(path, RemoveBags=False):
    '''
    :param path: directory which contains both raw and segmentation label images
    :return: seprate lists of file names for
    '''
    files = os.listdir(path)
    filesnames_raw,filesnames_labels = [],[]

    filesnames_raw = list(filter(lambda k: '_img.' in k, files))
    filesnames_labels = list(filter(lambda k: '_segmask.' in k, files))

    if RemoveBags:
      with open ("/data/people/tsingde/thesis/train_bags.json","r") as fp:
        bag_list = json.load(fp)
        for ele in bag_list:
          if ele+'_img.png' in filesnames_raw:
            filesnames_raw.remove(ele+'_img.png')
            filesnames_labels.remove(ele+'_segmask.png')

    filesnames_raw = [path + s  for s in filesnames_raw]
    filesnames_labels = [path + s  for s in filesnames_labels]

    filesnames_raw.sort()
    filesnames_labels.sort()

    return filesnames_raw,filesnames_labels


def random_crop(image):
    cropped_image = tf.image.random_crop(
        image, size=[1, img_height_width, img_height_width, image.shape[-1]])

    return cropped_image



def random_jitter(image):
  # resizing to 286 x 286 x 3
    image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
    image = random_crop(image)

  # random mirroring
    image = tf.image.random_flip_left_right(image)

    return image




def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image

def parse_function(files, labels):

  image_raw = tf.io.read_file(files)
  image_seg = tf.io.read_file(labels)

  image_raw =  tf.image.decode_png(image_raw, channels=1) 
  image_seg = tf.image.decode_png(image_seg, channels=1)



  image = tf.concat([ tf.cast(image_raw,dtype=tf.float32) , tf.cast(image_seg,dtype=tf.float32)],axis=2 )
  return image


def parse_function_sim(files, labels):
  image_raw = tf.io.read_file(files)
  image_seg = tf.io.read_file(labels)

  image_raw = tf.image.decode_png(image_raw, channels=1)
  image_seg = tf.image.decode_png(image_seg, channels=3)

  image = tf.concat([tf.cast(image_raw,dtype=tf.float32), tf.cast(image_seg,dtype=tf.float32)],axis=2 )
  return image


##-------------------------------------------- managing training processes --------------------------------------------


def limit_GPU(num):
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[num], 'GPU')
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)


def set_channels(ch0):
    global kept_channels
    global omitted_channels
    omitted_channels = [item for item in kept_channels if item not in ch0]
    kept_channels = ch0
    if len(omitted_channels):
        kept_channels.remove(0)



def encode(im, simulated=False, crop_flip=True):   
  global kept_channels
  global omitted_channels

  #im : B,H;W,C
  #simulated: a flag to handle pixel codes in simulated images
  im_raw = im[:,:,:,:1]  
  im_seg = im[:,:,:,1:] 

  
  if simulated:
    seg_sum = tf.math.reduce_sum(im_seg, 3)
    #This one-hot encoding logic depends on uniqueness of sum of seg_mask_rbg values
    
    background_channel =  tf.expand_dims( 
                                    tf.where( 
                                        tf.math.reduce_any(
                                                            tf.concat( [ tf.expand_dims(tf.equal(seg_sum, 0.),-1), 
                                                                         tf.expand_dims(tf.equal(seg_sum, 226.),-1), 
                                                                         tf.expand_dims(tf.equal(seg_sum, 380.),-1)] 
                                                                        , 3) , 
                                                            3), 
                                            1.0, 0.0) 
                                    ,3)

    box_channel =  tf.expand_dims( tf.where( tf.equal(seg_sum, 221.), 1.0, 0.0 ) , 3  )
    bag_channel = tf.expand_dims( 
                                    tf.where( 
                                        tf.math.reduce_any(
                                                            tf.concat( [ tf.expand_dims(tf.equal(seg_sum, 321.),-1), 
                                                                         tf.expand_dims(tf.equal(seg_sum, 501.),-1)] 
                                                                        , 3) , 
                                                            3), 
                                            1.0, 0.0) 
                                    ,3)
    
    strap_channel = tf.zeros_like(box_channel)
    tote_channel = tf.zeros_like(box_channel)
    wood_channel =  tf.expand_dims(tf.where( tf.equal(seg_sum, 187.), 1.0, 0.0 ) , 3  )
    other_channel = tf.zeros_like(box_channel)
    totehole_channel = tf.zeros_like(box_channel)

    im_onehot = tf.concat([background_channel,box_channel,bag_channel,strap_channel,tote_channel,wood_channel,other_channel,totehole_channel], 3)

  else:
    im_seg = im_seg/10
    im_seg = tf.cast(im_seg, tf.uint8)
    im_onehot = tf.squeeze(  tf.one_hot(im_seg, depth=8),  3)
  

  if len(omitted_channels):
    #All the channels that are removed, sementically becomes background 
    c0 = tf.expand_dims(tf.reduce_sum(tf.gather(im_onehot, omitted_channels +[0],axis=-1),3),-1)
    rest = tf.gather(im_onehot,kept_channels,axis=-1)
    im_onehot = tf.concat([c0,rest], -1)

  im = tf.concat([im_raw,im_onehot], axis=3 )
  
  #random crop-flip-resize  or  just resize 
  if crop_flip:
    im=random_jitter(im)
  else:
    im = tf.image.resize(im, [img_height_width, img_height_width])
  
  #Normalise
  im_raw = im[:,:,:,:1]  
  im_onehot = im[:,:,:,1:]
  im_raw = normalize(im_raw)
  im = tf.concat([im_raw,im_onehot], axis=3 )

  return im


def generate_images(model, test_input,time_path,epoch):
  test_input = encode(test_input,True, False)
  #test_input = test_input[:,:,:,:1]
  prediction = model(test_input)

  print_data(test_input[0],prediction[0],time_path,epoch,'')


def trained_check(G,time_path):
    with open(examples_file,'r') as f:
        emp_dict= json.load(f)
    im_nums = sum(emp_dict.values(), [])

    im_raws = [ data_sim + s + '_rgb.png'  for s in im_nums]
    im_segs = [ data_sim + s + '_seg_mask_rgb.png'  for s in im_nums]

    im_list = []
    for i in range(len(im_raws)):
        im_list.append(  tf.expand_dims(parse_function_sim(im_raws[i],im_segs[i]) , 0) )

    #batch =  tf.concat(im_list, 0)
    #ip = encode(batch,True,False)
    headings = ['empty_0','empty_1','empty_2',
                'box_0','box_1','box_2','box_3',
                'bag_0','bag_1','bag_2','bag_3',
                'bagbox_0','bagbox_1','bagbox_2','bagbox_3',
                'extra_0','extra_1'
                ]

    for i in range(len(headings)):
      ip = encode(im_list[i], True, False)
      #ip = ip[:,:,:,:1]
      op = G(ip)
      print_data(ip[0],op[0], time_path, 'trained_',headings[i])


def loss(l_dict,time_path):
    #---- save loss values ----
    save_train_res = '/data/people/tsingde/thesis/train/'
    path = os.path.join(save_train_res, time_path)
    path = os.path.join(path,'results/')
    os.makedirs(path, exist_ok=True)

    file = os.path.join(path, "loss_df.csv")
    cols = ['epoch',
            'gen G', 'total cycle_g', 'identity_y', 
            'gen f', 'total cycle_f', 'identity_x',
            'disc x', 'disc x gp',
            'disc y', 'disc y gp']
    loss_df = pd.DataFrame.from_dict(l_dict, orient='index', columns=cols)
    loss_df.to_csv(file,index=False)

    #---- create loss curves ----
    sns.set_style("whitegrid",{"grid.linestyle": ":"})
    fig, axs = plt.subplots(10, 1, figsize=(6,32))
    sns.lineplot( data=loss_df, x='epoch', y=cols[1] , ax=axs[0] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[2] , ax=axs[1] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[3] , ax=axs[2] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[4] , ax=axs[3] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[5] , ax=axs[4] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[6] , ax=axs[5] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[7] , ax=axs[6] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[8] , ax=axs[7] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[9] , ax=axs[8] )
    sns.lineplot( data=loss_df, x='epoch', y=cols[10] , ax=axs[9] )
    fig.tight_layout()

    save_fig(fig,time_path,'training_','loss_cruves')



#----------------- semantic loss -------------------------

def gray2rgb(im):
  # Creates RGB images out of grayscale images, by repeating the identical values in all 3 channels
  # (throws away the one-hot-encoded seg_mask)
  im_raw_gray = im[:,:,:,:1]
  im_raw_rgb =  tf.squeeze(tf.stack([ im_raw_gray, im_raw_gray, im_raw_gray],3) , -1)

  #resizes image to fit the pretrained models
  im_raw_rgb = tf.image.resize(im_raw_rgb, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  
  return im_raw_rgb

def load_pretrained_model(name='mobilenet'):
  # Loads an image classififcation model pretrained on imagenet. 
  # Cuts of some layers from the end, to give an output feature map of size (B,7,7,C)

    if name.lower()=='mobilenet':
        new_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=None, alpha=1.0, include_top=True, weights=None,
        input_tensor=None, pooling=None, classes=1000,
        classifier_activation='softmax')
        new_model.load_weights('/data/people/tsingde/thesis/pretrained/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224.h5')
        model2 = Model(new_model.input, new_model.layers[-6].output) #(b,7,7,320)

    elif name.lower()=='resnet':
        new_model=tf.keras.applications.resnet50.ResNet50(
        include_top=True, weights=None, input_tensor=None,
        input_shape=None, pooling=None, classes=1000)
        new_model.load_weights('/data/people/tsingde/thesis/pretrained/resnet50_weights_tf_dim_ordering_tf_kernels.h5')
        model2 = Model(new_model.input, new_model.layers[-7].output) #(b,7,7,512)

        
    return model2


#Have to create this model globally, for could not be called inside a function with @tf.function decorater
def set_pretrained_model(flag='mobilenet'):
  global pretrained_model
  pretrained_model = load_pretrained_model(flag)


def reduce_featuremap_size(feature_map, option):
    if option.lower()=='depthpooling':
        #does max pooling across all the channels
        small_feature = tf.math.reduce_max(feature_map, 3, True) #(b,7,7,1) 
    elif option.lower()=='maxpooling':
        small_feature = tf.nn.max_pool2d(feature_map, ksize=(7,7), strides=1, padding='VALID')  #(b,1,1,c)  
    elif option.lower()=='none':
        small_feature = feature_map
    return small_feature


def semantic_loss(real,fake, reducefeaturemap_option='maxpooling' ):
    global pretrained_model

    real = gray2rgb(real) 
    fake = gray2rgb(fake)

    real_feature = reduce_featuremap_size( pretrained_model(real) , reducefeaturemap_option)
    fake_feature = reduce_featuremap_size( pretrained_model(fake) , reducefeaturemap_option)

    #L1 Loss
    loss = tf.reduce_mean(tf.abs(real_feature - fake_feature))
    
    return loss
