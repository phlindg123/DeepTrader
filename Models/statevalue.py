

import tensorflow as tf
import tensorflow.keras as k

class VNetwork:
    def __init__(self, state_size, action_size,alpha,sess):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.sess = sess

        k.backend.set_session(sess)

        self.model, self.weights = self.create_network()
    def create_network(self):
        state_prices = k.layers.Input(shape=(self.state_size), name="Price_input")
        state_portfolio = k.layers.Input(shape=(self.action_size), name="Portfolio_inpit")
        s1 = k.layers.Dense(256, activation="relu", name="dense_price")(state_prices)
        p1 = k.layers.Dense(256, activation="relu", name="dense_port")(state_portfolio)
        concat = k.layers.concatenate([s1,p1], axis=1)
        s2 = k.layers.Dense(128, activation="relu", name="dense2")(concat)
        he_normal = tf.initializers.he_normal()
        s2 = k.layers.Flatten()(s2)
        output = k.layers.Dense(1, name="value_output", kernel_initializer=he_normal)(s2)

        model = k.Model(inputs = [state_prices, state_portfolio], outputs = output)
        adam = k.optimizers.Adam(lr = self.alpha)
        model.compile(loss="mse", optimizer=adam)
        #print(model.summary())
        return model, model.trainable_weights