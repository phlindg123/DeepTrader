
import tensorflow as tf
import tensorflow.keras as k


class ActorNetwork:
    def __init__(self, state_size, action_size, tau, sess):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = tau
        self.sess = sess
        
        k.backend.set_session(sess)

        self.model, self.weights, self.state = self.create_network()
        self.target_model, self.target_weights, self.target_state = self.create_network()

        self.action_gradient = tf.placeholder(tf.float32, [None, self.action_size[0]])
        self.params_grad = tf.gradients(self.model.output, self.weights, -self.action_gradient)
        grads = zip(self.params_grad, self.weights)
        self.optimize = tf.train.AdamOptimizer(1e-3).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())
    def train(self, states, action_grads):
        self.sess.run(self.optimize, feed_dict= {
            self.state: states,
            self.action_gradient: action_grads,
        })

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(weights)):
            target_weights[i] = self.tau*weights[i] + (1 - self.tau)*target_weights[i]
        self.target_model.set_weights(target_weights)
        
    def create_network(self):
        state = k.layers.Input(shape=(self.state_size), name="actor_state_input")

        random_uniform = tf.initializers.random_uniform()
        trunc_normal = k.initializers.TruncatedNormal()
        s1 = k.layers.Dense(256, name="actor_dense1", activation="relu")(state)
        #s2 = k.layers.GaussianNoise(1.0)(s1)
        s3 = k.layers.Dense(128, name="actor_dense2", activation="relu")(s1)
        #s4 = k.layers.GaussianNoise(1.0)(s3)
        output = k.layers.Dense(self.action_size[0], activation="tanh", name="actor_output", kernel_initializer=trunc_normal)(s3)

        model = k.Model(inputs = state, outputs=output)
        print(model.summary())
        return model, model.trainable_weights, state