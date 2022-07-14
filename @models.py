import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, MaxPooling2D, Dropout, Flatten

import time

def get_dropout(input_tensor, p=0.3, mc=False):
    if mc: 
        layer = Dropout(p, name='top_dropout')
        return layer(input_tensor, training=True)
    else:
        return Dropout(p, name='top_dropout')(input_tensor, training=False)

def run_expriment(model_name, train_dataset, val_dataset, n_res, n_classes, batch_size=32, mc=False, epochs=100): 
    
    if model_name == 'efficient':
        # base_model = keras.applications.EfficientNetB0(include_top=False, input_shape=(N_RES, N_RES, 3),  weights = 'imagenet')
        base_model = keras.applications.EfficientNetB7(include_top=False, input_shape=(n_res, n_res, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(n_res, n_res, 3))
        x = base_model(inputs)
        x = keras.layers.GlobalAveragePooling2D()(x) 
        x = get_dropout(x, mc)
        x = keras.layers.Dense(n_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        
    # VGG16 
    else:
        base_model = keras.applications.VGG16(include_top=False, input_shape=(n_res, n_res, 3),  weights = 'imagenet')
        base_model.trainable = True
        
        inputs = keras.Input(shape=(n_res, n_res, 3))
        x = base_model(inputs)
        x = keras.layers.Flatten(name = "avg_pool")(x) 
        x = keras.layers.Dense(512, activation='relu')(x)
        x = get_dropout(x, mc)
        x = keras.layers.Dense(256, activation='relu')(x)
        x = keras.layers.Dense(n_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=inputs, outputs=x)
        

    sv = [tf.keras.callbacks.ModelCheckpoint(os.path.join(f'models/{model_name}_mc-{str(mc)}_bs-{batch_size}_{time.strftime("%Y%m%d-%H%M%S")}.h5'), 
                                             monitor='val_accuracy', 
                                             verbose=0, 
                                             save_best_only=True,
                                             save_weights_only=True, 
                                             mode='max', 
                                             save_freq='epoch'), 
          tf.keras.callbacks.EarlyStopping(monitor = 'val_accuracy', 
                                           patience = 4, 
                                           min_delta = 0.01)
          ]

    
    LR = 0.0001
    # steps_per_epoch = len(train_images) // batch_size
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(LR, steps_per_epoch*30, 0.1, True)
    
    # sgd = tf.keras.optimizers.SGD(0.01)
    # moving_avg_sgd = tfa.optimizers.MovingAverage(sgd)
    
    
    model.compile(loss='sparse_categorical_crossentropy', 
                #   optimizer = moving_avg_sgd, 
                  optimizer = tf.keras.optimizers.Adam(LR), 
                #   optimizer = tf.keras.optimizers.Adam(lr_schedule), 
                  metrics=['accuracy'])
    
    hist = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=epochs,
                    # class_weight=class_weights, 
                    verbose=1,
                    shuffle=True,
                    callbacks=[sv])
    
    # histories.append(hist)
    
    return model, hist