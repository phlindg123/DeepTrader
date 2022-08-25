

import tensorflow as tf
import tensorflow.python.keras as k

class CriticNetwork:
    def __init__(self, state_size, action_size, tau, sess):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.sess = sess
        
        k.backend.set_session(sess)

        self.model, self.action, self.state = self.create_network()
        self.target_model, self.target_action, self.target_state = self.create_network()
        self.action_grads = tf.gradients(self.model.output, self.action)
        self.sess.run(tf.global_variables_initializer())


    def gradients(self, states, actions):
        return self.sess.run(self.action_grads, feed_dict={
            self.state: states,
            self.action : actions
        })

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau*weights[i] + (1 - self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)

    def create_network(self):
        state = k.layers.Input(shape=(self.state_size), name="critic_state_input")
        action = k.layers.Input(shape=(self.action_size), name="critic_action_input")
        he_normal = tf.initializers.he_normal()
        s1 = k.layers.Dense(256, activation="relu", name="critic_dense1")(state)
        x = k.layers.concatenate([s1, action], axis=-1)
        c1 = k.layers.Dense(128, activation="relu", name="critic_combined_dens1")(x)
       # c1 = k.layers.Dropout(0.1)(c1)
        
        output = k.layers.Dense(1, name="critic_combined_dens2",activation="linear", kernel_initializer=he_normal)(c1)
        model = k.Model(inputs = [state,action], outputs = output)
        adam = k.optimizers.Adam(lr=1e-3)
        model.compile(loss="mse", optimizer=adam)
        print(model.summary())
        return model, action, state

    