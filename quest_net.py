import tensorflow as tf
from utils import*


class QuestNet(object):
    def __init__(self, input_dim, num_quests,hidden_dim=4,dq=4, dk=4, name=None, act=tf.nn.relu, learning_rate=0.01):
        super(QuestNet, self).__init__()
        self.input_dim = input_dim
        self.name = name
        self.hidden_dim = hidden_dim
        self.act = act
        self.dk = dk
        self.dq = dq
        self.w1 = glorot([self.input_dim, self.hidden_dim])
        self.b1 = zeros([self.hidden_dim])
        self.w2 = glorot([self.hidden_dim, 4])
        self.b2 = zeros([4])
        self.qw = []
        for n in num_quests:
            self.qw.append(glorot([dq,n]))
        # for attention
        self.theta1 = glorot([self.dk,3*self.dq])
        self.theta2 = glorot([4, self.dk])

    def __call__(self, x, A, slices):
        # graph embedding
        h1_1 = tf.matmul(A, x)
        h1_1 = h1_1@ self.w1
        h1_hat = h1_1 + self.b1
        h1 = self.act(h1_hat)
        h2_hat = tf.matmul(tf.matmul(A, h1), self.w2) + self.b2
        h2 = self.act(h2_hat)

        # slice features
        sf=[]
        for i in range(len(slices)):
            sf.append( self.qw[i] @ slices[i]   )

        sf=tf.concat(sf,axis=1)
        # attention
        q = tf.matmul(self.theta1,sf)
        k = tf.matmul(h2, self.theta2)
        # qt = tf.transpose(q)
        u = tf.matmul(k,q)

        w = tf.nn.softmax(u, axis=1)

        return w


def loss(pred, y):
    return tf.reduce_mean(tf.square(tf.subtract(pred, y)))


def train(model, inputs, outputs, learning_rate):
    x, A, quest = inputs
    # auto gradient
    with tf.GradientTape() as t:
        current_loss = loss(model(x, A, quest), outputs)
    dw1, db1, dw2, db2, dt1, dt2 = t.gradient(current_loss,
                                                   [model.w1, model.b1, model.w2, model.b2 , model.theta1,
                                                    model.theta2])
    model.w1.assign_sub(learning_rate * dw1)
    model.b1.assign_sub(learning_rate * db1)
    model.w2.assign_sub(learning_rate * dw2)
    model.b2.assign_sub(learning_rate * db2)
    model.theta1.assign_sub(learning_rate * dt1)
    model.theta2.assign_sub(learning_rate * dt2)

    with tf.GradientTape() as t:
        current_loss = loss(model(x, A, quest), outputs)
        dqw=t.gradient(current_loss,model.qw)
    for i in range(len(dqw)):
        model.qw[i].assign_sub(learning_rate *dqw[i])