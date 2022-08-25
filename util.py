


import numpy as np
from keras.models import model_from_config, Sequential, Model
import keras.optimizers as optimizers
import keras.backend as K
import tensorflow as tf

def clone_model(model, custom_objects={}):
    config = {
        "class_name": model.__class__.__name__,
        "config": model.get_config()
    }
    clone = model_from_config(config, custom_objects=custom_objects)
    clone.set_weights(model.get_weights())
    return clone

def get_soft_target_model_updates(target, source, tau):
    target_weights = target.trainable_weights + sum([l.non_trainable_weights for l in target.layers], [])
    source_weights = source.trainable_weights + sum([l.non_trainable_weights for l in source.layers], [])
    assert len(target_weights) == len(source_weights)

    # Create updates.
    updates = []
    for tw, sw in zip(target_weights, source_weights):
        updates.append((tw, tau * sw + (1. - tau) * tw))
    return updates

class AdditionalUpdatesOptimizer(optimizers.Optimizer):
    def __init__(self, optimizer, additional_updates):
        super(AdditionalUpdatesOptimizer, self).__init__()
        self.optimizer = optimizer
        self.additional_updates = additional_updates

    def get_updates(self, params, loss):
        updates = self.optimizer.get_updates(params=params, loss=loss)
        updates += self.additional_updates
        self.updates = updates
        return self.updates

    def get_config(self):
        return self.optimizer.get_config()

def huber_loss(y_true, y_pred, clip_value):
    assert clip_value > 0

    x = y_true - y_pred
    if np.isinf(clip_value):
        return 0.5*K.square(x)
    condition = K.abs(x) < clip_value
    squared_loss = 0.5* K.square(x)
    linear_loss = clip_value * (K.abs(x) - 0.5*clip_value)
    if hasattr(tf, "select"):
        return tf.select(condition, squared_loss, linear_loss)
    else:
        return tf.where(condition, squared_loss, linear_loss)