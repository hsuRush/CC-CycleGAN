

import numpy as np
from argparse import ArgumentParser
import keras
from keras.models import Model
from keras.layers import GRU, CuDNNGRU, Bidirectional
from keras.layers import Input, Dense, Lambda, Reshape, BatchNormalization, Activation
from keras.layers.merge import add, concatenate
from keras import backend as K

try:
    import para
except ImportError:
    try:
        from . import para
    except ImportError:
        raise ImportError("import para error")
    else:
        pass
else:
    pass

try:
    import resnet
except ImportError:
    try:
        from . import resnet
    except ImportError:
        raise ImportError("import resnet error")
    else:
        pass
else:
    pass
# input_shape = (h,w,c)
N_COL = para.image_h
N_ROW = para.image_w

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def build_ctc_network(args, training=True):
    global N_COL
    global N_ROW
    
    """
    if args.stn == True:
        from STN.STN import locnet
        input_tensor =  locnet(input, sampling_size = (N_COL, N_ROW))
        print("STN")
    """
    #stn_model = None
    if hasattr(args, 'stn_model'):
        pass
    else:
        args.stn_model = False

    if args.model == 'resnet18':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    elif args.model == 'resnet18_2222':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18_2222(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    elif args.model == 'resnet18_2222_64':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18_2222_start_from64(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    elif args.model == 'resnet18_2222_48':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18_2222_start_from48(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    elif args.model == 'resnet18_2222_32':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18_2222_start_from32(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    elif args.model == 'resnet18_2222_16':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_18_2222_start_from16(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args) 
    elif args.model == 'resnet34':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_34(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    elif args.model == 'resnet50':
        NOT_CARE = 1
        base_model = resnet.ResnetBuilder.build_resnet_50(input_shape=(N_COL,N_ROW,3), num_outputs=NOT_CARE, include_top=False, args=args)
    else:
        raise TypeError('model should be in the list of the supported model!')

    #print('Input col: ', N_COL)
    #print('Input row: ', N_ROW)

    x = base_model.output
    #CNN to RNN
    x = Lambda(lambda x: K.permute_dimensions(x,(0,2,1,3)))(x) # switchaxes from [b,h,w,c] to [b,w,h,c]
    conv_shape = x.get_shape() # b, h,w,c  resnet 18 -> (?, 8, 16, 256)
    #print('conv_shape', conv_shape)
    x = Reshape(target_shape=(int(conv_shape[1]), int(conv_shape[2]*conv_shape[3])), name='reshape')(x) #(?, 16, 8, 256)
    x = Dense(para.dense_size, activation='relu', kernel_initializer='he_normal', name='dense1')(x) #5_exp ->128
    #x = BatchNormalization()(x)
    # GRU RNN

    GRU_UNIT = CuDNNGRU
    
    bi_gru1 = Bidirectional(GRU_UNIT(para.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='bi_gru1'), merge_mode='sum')(x)
    bi_gru1 = BatchNormalization()(bi_gru1)
    
    bi_gru2 = Bidirectional(GRU_UNIT(para.rnn_size, return_sequences=True, kernel_initializer='he_normal', name='bi_gru2'), merge_mode='concat')(bi_gru1)
    bi_gru2 = BatchNormalization()(bi_gru2)
    
    #attention
    #att = AttentionWithContext()(bi_gru2)
    inner = Dense(para.num_classes, kernel_initializer='he_normal',name='dense2')(bi_gru2)
    y_pred = Activation('softmax', name='softmax')(inner)

    labels = Input(name='the_labels', shape=[para.max_text_len], dtype='float32') # (None ,7)
    input_length = Input(name='input_length', shape=[1], dtype='int64')     # (None, 1)
    label_length = Input(name='label_length', shape=[1], dtype='int64')     # (None, 1)

    # Keras doesn't currently support loss funcs with extra parameters
    # so CTC loss is implemented in a lambda layer
    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length]) #(None, 1)

    if training:
        return Model(inputs=[base_model.input, labels, input_length, label_length], outputs=loss_out), conv_shape[1]
    else:
        return Model(inputs=[base_model.input], outputs=y_pred)