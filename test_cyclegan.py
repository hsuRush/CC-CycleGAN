from __future__ import print_function, division
import scipy

from keras.datasets import mnist
from instanceNormalization import InstanceNormalization
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate, GRU, CuDNNGRU, Bidirectional, Lambda
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.merge import add, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import backend as K

import datetime

import matplotlib
#matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import sys
from data_loader import DataLoader
from options import train_options, test_options
import para
import json

import numpy as np
import os
from os.path import join

class CycleGAN():
    def __init__(self, args):
        # Input shape
        self.args = args
        self.img_rows = 120#128
        self.img_cols = 240#128
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.phase = 'test' if args.test else 'train'
        # Configure data loader
        self.dataset_name = args.dataset
        self.data_loader = DataLoader(dataset_name=self.dataset_name,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch_rows = int(self.img_rows / 2**3)
        patch_cols = int(self.img_cols / 2**3)
        self.disc_patch = (patch_rows, patch_cols, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss
        self.lambda_condition = self.lambda_cycle * .1

        self.lr = 2e-4

        self.args_append_attr() 

        #@TODO load args.json and overwrite the default
        #with open(join(args.exp_dir, 'args.json'), 'r') as f:
        #    json.dump(vars(args), f, ensure_ascii=False, indent=2, sort_keys=True)

        #optimizer = Adam(self.lr)

         # Build ctc net
        if args.ctc_condition:
            self.ctc_model = self.build_condition_network(training=True, condition='ctc')
            if self.args.verbose:
                print('------------ctc-model-----------')
                self.ctc_model.summary()
        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        if self.args.verbose:
            print('------------d_A-----------')
            self.d_A.summary()
            print('------------d_B-----------')
            self.d_B.summary()
        #self.d_A.compile(loss='mse',
        #    optimizer=Adam(self.lr/2),
        #    metrics=['accuracy'])
        #self.d_B.compile(loss='mse',
        #    optimizer=Adam(self.lr/2),
        #    metrics=['accuracy'])
        #self.ctc_model.compile(optimizer=optimizer, loss={'ctc': lambda y_true, y_pred: y_pred})
        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------
        
       

        # Build the generators
        self.g_AB = self.build_generator()
        self.g_BA = self.build_generator()

        self.load_model(self.args.exp_dir, self.args.resume_epoch)
        
        
    def args_append_attr(self):
        self.args.lr = self.lr
        self.args.img_rows = self.img_rows
        self.args.img_cols = self.img_cols
        self.args.channels = self.channels 
        self.args.img_shape = self.img_shape
        self.args.phase = self.phase
        
        self.args.dataset_name = self.dataset_name
        self.args.disc_patch = self.disc_patch

        # Number of filters in the first layer of G and D
        self.args.gf = self.gf
        self.args.df = self.df

        # Loss weights
        self.args.lambda_cycle = self.lambda_cycle   
        self.args.lambda_id = self.lambda_id  
        self.args.lambda_condition = self.lambda_condition

    def load_model(self, exp_dir, resume_epoch):
        if self.args.ctc_condition:
            self.ctc_model.load_weights(join(exp_dir,    'ctc_weights_{}.h5').format(resume_epoch))
        self.d_A.load_weights(join(exp_dir,          'd_A_weights_{}.h5').format(resume_epoch))
        self.d_B.load_weights(join(exp_dir,          'd_B_weights_{}.h5').format(resume_epoch))
        self.g_AB.load_weights(join(exp_dir,         'g_AB_weights_{}.h5').format(resume_epoch))
        self.g_BA.load_weights(join(exp_dir,         'g_BA_weights_{}.h5').format(resume_epoch))
        #self.combined.load_weights(join(exp_dir, 'combined_weights_{}.h5').format(resume_epoch))

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        d0 = Input(shape=self.img_shape)

        # Downsampling
        #d1 = conv2d(d0, self.gf)
        d2 = conv2d(d0, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        #u3 = deconv2d(u2, d1, self.gf)

        u3 = UpSampling2D(size=2)(u2)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u3)

        return Model(d0, output_img)

    def build_discriminator(self):
        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        #d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(img, self.df*2, normalization=False)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

        return Model(img, validity)

    def build_condition_network(self, training='train', condition='ctc'):
        if condition == 'ctc':
            from model import build_ctc_network
            bool_training = True if training == 'train' else False
            return build_ctc_network(self.args, training=bool_training)

    def test_A2B(self, batch_size=1, iteration=64, set='test', save_dir='test_images'):
        assert args.test == True
        save_dir = join(save_dir, self.dataset_name)
        #os.makedirs(join(save_dir, '%s/comparison/%s' % (self.dataset_name, set)), exist_ok=True)
        os.makedirs(join(save_dir, '%s/transfered_image_A2B/%s' % (args.exp_dir, set)), exist_ok=True)
        
        from tqdm import tqdm
        for batch_i, (imgs_A, lbl_A) in enumerate(tqdm(self.data_loader.load_batch_A(batch_size=batch_size, set=set, is_testing=True, iteration=iteration, condition=True))):
            
            

            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            #fake_A = self.g_BA.predict(imgs_B)

            #reconstr_B = self.g_AB.predict(fake_A)
            reconstr_A = self.g_BA.predict(fake_B)
            # Save transfered image
            self.saved_transfered_image(fake_B, None, lbl_A, None, save_dir=join(save_dir, '%s/transfered_image_A2B/%s' % (args.exp_dir, set)), set=set, batch_id=batch_i)


    def test_B2A(self, batch_size=1, iteration=64, set='test', save_dir='test_images'):
        assert args.test == True
        save_dir = join(save_dir, self.dataset_name)
        #os.makedirs(join(save_dir, '%s/comparison/%s' % (self.dataset_name, set)), exist_ok=True)
        os.makedirs(join(save_dir, '%s/transfered_image_B2A/%s' % (args.exp_dir, set)), exist_ok=True)
        
        from tqdm import tqdm      
        for batch_i, (imgs_B, lbl_B) in enumerate(tqdm(self.data_loader.load_batch_B(batch_size=batch_size, set=set, is_testing=True, iteration=iteration, condition=True))):

            # Translate images to opposite domain
            #fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            reconstr_B = self.g_AB.predict(fake_A)
            #reconstr_A = self.g_BA.predict(fake_B)
            # Save transfered image
            self.saved_transfered_image(None, fake_A, None, lbl_B, save_dir=join(save_dir, '%s/transfered_image_B2A/%s' % (args.exp_dir, set)), set=set, batch_id=batch_i)

                

    def test_both(self, batch_size=1, iteration=64, set='test', save_dir='test_images'):
        assert args.test == True
        save_dir = join(save_dir, self.dataset_name)
        os.makedirs(join(save_dir, '%s/comparison/%s' % (args.exp_dir, set)), exist_ok=True)
        #os.makedirs(join(save_dir, '%s/transfered_image/%s' % (args.exp_dir, set)), exist_ok=True)
        
        from tqdm import tqdm
        for batch_i, (imgs_A, imgs_B ,lbl_A, lbl_B) in enumerate(tqdm(self.data_loader.load_batch(batch_size=batch_size, set=set, is_testing=True, iteration=iteration, condition=True))):

            # Translate images to opposite domain
            fake_B = self.g_AB.predict(imgs_A)
            fake_A = self.g_BA.predict(imgs_B)

            reconstr_B = self.g_AB.predict(fake_A)
            reconstr_A = self.g_BA.predict(fake_B)
            # Save transfered image
            #self.saved_transfered_image(fake_B, fake_A, lbl_A, lbl_B, save_dir=save_dir, set=set, batch_id=batch_i)
            
            # Comparison result 
            gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])
            # Rescale images 0 - 1
            gen_imgs = 0.5 * gen_imgs + 0.5

            titles = ['Original', 'Translated', 'Reconstructed']
            r, c = 2, 3
            fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    axs[i,j].imshow(gen_imgs[cnt])
                    axs[i, j].set_title(titles[j])
                    axs[i,j].axis('off')
                    cnt += 1
            fig.savefig(join(save_dir, "%s/comparison/%s/%d.png" % (args.exp_dir, set, batch_i)))
            plt.close()
                # Comparison result 
               

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        imgs_A = self.data_loader.load_data(domain="A", batch_size=1, is_testing=True)
        imgs_B = self.data_loader.load_data(domain="B", batch_size=1, is_testing=True)

        # Demo (for GIF)
        #imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        #imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt])
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

    def save_model(self, epochs, exp_dir):
        
        if self.args.ctc_condition:
            self.ctc_model.save_weights(join(exp_dir,'ctc_weights_{}.h5').format(epochs))
        self.d_A.save_weights(join(exp_dir,      'd_A_weights_{}.h5').format(epochs))
        self.d_B.save_weights(join(exp_dir,      'd_B_weights_{}.h5').format(epochs))
        self.g_AB.save_weights(join(exp_dir,     'g_AB_weights_{}.h5').format(epochs))
        self.g_BA.save_weights(join(exp_dir,     'g_BA_weights_{}.h5').format(epochs))
        self.combined.save_weights(join(exp_dir,'combined_weights_{}.h5').format(epochs))

    def saved_transfered_image(self, fake_B, fake_A, lbl_A, lbl_B, save_dir, set, batch_id):
        """
        fake_B is from imageA send into GAB (so the label is A)
        fake)A is from imageB send into GBA (so the label is B)
        """
        path = save_dir #join(save_dir, '%s/transfered_image/%s' % (self.dataset_name, set))
        import cv2 

        os.makedirs(path, exist_ok=True)
        #print(type(lbl_A), type(fake_B))
        #print(fake_B.shape)
        for b in range(args.batch):
            image_id = batch_id * args.batch + b 
            if isinstance(fake_B, np.ndarray) & isinstance(lbl_A, list):
                cv2.imwrite(join(path, '%d_%s.png'%(image_id, lbl_A[b])), self.unnormalize(fake_B[b]))
                print('wrote to ', join(path, '%d_%s.png'%(image_id, lbl_A[b])))
            if isinstance(fake_A, np.ndarray) & isinstance(lbl_B, list):
                cv2.imwrite(join(path, '%d_%s.png'%(image_id, lbl_B[b])), self.unnormalize(fake_A[b]))
                print('wrote to ', join(path, '%d_%s.png'%(image_id, lbl_B[b])))

    def unnormalize(self, im):
        #im = np.array(im)
        im = np.array(255 * (0.5 * im + 0.5), dtype=np.uint8)
        #print(im.shape)
        #print(im)
        
        return im
        
if __name__ == '__main__':
    args = test_options()
    gan = CycleGAN(args)
    if args.direction == 'both':
        gan.test_both(batch_size=args.batch, iteration=args.iteration, set=args.set)
    elif args.direction == 'A2B':
        gan.test_A2B(batch_size=args.batch, iteration=args.iteration, set=args.set)
    elif args.direction == 'B2A':
        gan.test_B2A(batch_size=args.batch, iteration=args.iteration, set=args.set)