import scipy
from glob import glob
import numpy as np
import os

max_text_len = 7 
CHAR_VECTOR = "abcdefghijklmnopqrstuvwxyz0123456789_" 
letters = [letter for letter in CHAR_VECTOR]
num_classes = len(letters)

class DataLoader():
    def __init__(self, dataset_name, img_res=(128, 128),):
        self.dataset_name = dataset_name
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, condition=False, is_testing=False):
        data_type = "train%s" % domain if not is_testing else "test%s" % domain
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        lbls = []
        lbl = None
        for img_path in batch_images:
            img = self.imread(img_path)
            if condition:
                lbl = os.path.basename(img_path).split('_')[1].split('.')[0] # xxx_???_xxx.png ???是車牌
            if not is_testing:
                img = scipy.misc.imresize(img, self.img_res)

                if np.random.random() > 0.5:
                    img = np.fliplr(img)
            else:
                img = scipy.misc.imresize(img, self.img_res)
            imgs.append(img)
            lbls.append(lbl)

        imgs = np.array(imgs)/127.5 - 1.
        
        if condition:
            return imgs, self._encode_batch_lbl(lbls)
        else:
            return imgs
    
    def load_batch(self, batch_size=1, set='train', is_testing=False, iteration=0, condition=False, labels_smoothing_epilson=0.0):
        assert (set =='train' or set == 'test')
        # labels_smoothing_epilson only activate if 'condition == True'

        #data_type = "train" if not is_testing else "test"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, set))
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, set))

        total_samples = None
        if iteration == 0: # default 
            self.n_batches = int(min(len(path_A), len(path_B)) / batch_size) 
            total_samples = self.n_batches * batch_size

        else:
            # check if more than the entire dataset
            if iteration > int(min(len(path_A), len(path_B)) / batch_size):
                print('iterations * batch_size > the number of dataset')
                iteration = int(min(len(path_A), len(path_B)) / batch_size)

            self.n_batches = iteration
            total_samples = self.n_batches * batch_size
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        path_B = np.random.choice(path_B, total_samples, replace=False)
        
            
        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, lbls_A, lbls_B = [], [], [], []
            lbl_A, lbl_B = None, None

            for img_A, img_B in zip(batch_A, batch_B):
                #print(img_A, img_B )
                if condition:
                    lbl_A = os.path.basename(img_A).split('_')[1].split('.')[0] # xxx_???_xxx.png ???是車牌
                    lbl_B = os.path.basename(img_B).split('_')[1].split('.')[0] # xxx_???_xxx.png ???是車牌
                    #print(lbl_A, lbl_B )
                img_A = self.imread(img_A)
                img_B = self.imread(img_B)
                ##condition
                

                img_A = scipy.misc.imresize(img_A, self.img_res)
                img_B = scipy.misc.imresize(img_B, self.img_res)

                #if not is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)
                if condition:
                    lbls_A.append(lbl_A)
                    lbls_B.append(lbl_B)
                    
            imgs_A = np.array(imgs_A)/127.5 - 1.
            imgs_B = np.array(imgs_B)/127.5 - 1.
            if condition and (not is_testing):
                yield imgs_A, imgs_B, self._encode_batch_lbl(lbls_A, labels_smoothing_epilson), self._encode_batch_lbl(lbls_B, labels_smoothing_epilson)
            
            elif condition and is_testing:
                yield imgs_A, imgs_B, lbls_A, lbls_B
            
            elif not condition:    
                yield imgs_A, imgs_B

    def load_batch_A(self, batch_size=1, set='train', is_testing=False, iteration=0, condition=False, labels_smoothing_epilson=0.0):
        assert (set =='train' or set == 'test')
        # labels_smoothing_epilson only activate if 'condition == True'

        #data_type = "train" if not is_testing else "test"
        path_A = glob('./datasets/%s/%sA/*' % (self.dataset_name, set))

        total_samples = None
        if iteration == 0: # default 
            self.n_batches = int(len(path_A) / batch_size) 
            total_samples = self.n_batches * batch_size

        else:
            # check if more than the entire dataset
            if iteration > int(len(path_A) / batch_size):
                print('iterations * batch_size > the number of dataset')
                iteration = int(len(path_A) / batch_size)

            self.n_batches = iteration
            total_samples = self.n_batches * batch_size
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_A = np.random.choice(path_A, total_samples, replace=False)
        
        for i in range(self.n_batches-1):
            batch_A = path_A[i*batch_size:(i+1)*batch_size]
            imgs_A, imgs_B, lbls_A, lbls_B = [], [], [], []
            lbl_A = None

            for img_A in batch_A:
                #print(img_A, img_B )
                if condition:
                    lbl_A = os.path.basename(img_A).split('_')[1].split('.')[0] # xxx_???_xxx.png ???是車牌
                    
                
                img_A = self.imread(img_A)
                ##condition
                

                img_A = scipy.misc.imresize(img_A, self.img_res)
                #if not is_testing and np.random.random() > 0.5:
                #        img_A = np.fliplr(img_A)
                #        img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                
                if condition:
                    lbls_A.append(lbl_A)
                    
                    
            imgs_A = np.array(imgs_A)/127.5 - 1.
            if condition and (not is_testing):
                yield imgs_A, self._encode_batch_lbl(lbls_A, labels_smoothing_epilson)
            
            elif condition and is_testing:
                yield imgs_A, lbls_A
            
            elif not condition:    
                yield imgs_A

    def load_batch_B(self, batch_size=1, set='train', is_testing=False, iteration=0, condition=False, labels_smoothing_epilson=0.0):
        assert (set =='train' or set == 'test')
        # labels_smoothing_epilson only activate if 'condition == True'

        #data_type = "train" if not is_testing else "test"
        path_B = glob('./datasets/%s/%sB/*' % (self.dataset_name, set))

        total_samples = None
        if iteration == 0: # default 
            self.n_batches = int(len(path_B) / batch_size) 
            total_samples = self.n_batches * batch_size

        else:
            # check if more than the entire dataset
            if iteration > int(len(path_B) / batch_size):
                print('iterations * batch_size > the number of dataset')
                iteration = int(len(path_B) / batch_size)

            self.n_batches = iteration
            total_samples = self.n_batches * batch_size
        # Sample n_batches * batch_size from each path list so that model sees all
        # samples from both domains
        path_B = np.random.choice(path_B, total_samples, replace=False)
        
        for i in range(self.n_batches-1):
            batch_B = path_B[i*batch_size:(i+1)*batch_size]
            imgs_B, imgs_B, lbls_B, lbls_B = [], [], [], []
            lbl_B = None

            for img_B in batch_B:
                #print(img_B, img_B )
                if condition:
                    lbl_B = os.path.basename(img_B).split('_')[1].split('.')[0] # xxx_???_xxx.png ???是車牌
                    
                
                img_B = self.imread(img_B)
                ##condition
                

                img_B = scipy.misc.imresize(img_B, self.img_res)
                #if not is_testing and np.random.random() > 0.5:
                #        img_B = np.fliplr(img_B)
                #        img_B = np.fliplr(img_B)

                imgs_B.append(img_B)
                
                if condition:
                    lbls_B.append(lbl_B)
                    
                    
            imgs_B = np.array(imgs_B)/127.5 - 1.
            if condition and (not is_testing):
                yield imgs_B, self._encode_batch_lbl(lbls_B, labels_smoothing_epilson)
            
            elif condition and is_testing:
                yield imgs_B, lbls_B
            
            elif not condition:    
                yield imgs_B

    def load_img(self, path):
        img = self.imread(path)
        img = scipy.misc.imresize(img, self.img_res)
        img = img/127.5 - 1.
        return img[np.newaxis, :, :, :]

    def imread(self, path):
        return scipy.misc.imread(path, mode='RGB').astype(np.float)
    
    def _encode_batch_lbl(self, batch_lbl, labels_smoothing_epilson=0):
        batch_de_lbls = []
        for lbl in batch_lbl:
            en_lbl = self._encode_lbl(lbl, labels_smoothing_epilson)
            dc_lbl = self._decodePlateVec(en_lbl)
            batch_de_lbls.append(dc_lbl)
        
        return batch_de_lbls

    def _decodePlateVec(self, y):
        vec = np.zeros((max_text_len), dtype=np.uint8)
        for i in range(7):
            vec[i] = np.argmax(y[:,i])
        return vec

    def _encode_lbl(self, string, labels_smoothing_epilson=0):
        #Applies label smoothing. See 5.4 and https://arxiv.org/abs/1512.00567
        num = np.zeros((num_classes, max_text_len))
        
        for i in range(len(string)):
            for j in range(num_classes):

                if ( string[i] == letters[j] ):
                    num[j,i] = 1
                    
        if (len(string) == 6):
            num[num_classes-1, 6] = 1
        if (len(string) == 5):
            num[num_classes-1, 6] = 1
            num[num_classes-1, 5] = 1
        if (len(string) == 4):
            num[num_classes-1, 6] = 1
            num[num_classes-1, 5] = 1
            num[num_classes-1, 4] = 1
        
        if labels_smoothing_epilson > 0.:
            num = num * (1-labels_smoothing_epilson) + (labels_smoothing_epilson / num_classes)    
        return num


if __name__ == "__main__":
    
    img_rows = 240#128
    img_cols = 120#128
    dataset_name = 'lpgen2aolp'
    data_loader = DataLoader(dataset_name=dataset_name,
                                      img_res=(self.img_rows, self.img_cols))

    for batch_i, data in enumerate(data_loader.load_batch(4, condition=True)):
        pass