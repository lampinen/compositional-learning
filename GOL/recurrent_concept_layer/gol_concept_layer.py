import numpy
import tensorflow as tf
import matplotlib.pyplot as plot


#Network = input -> *W1 + b1 -> relu -> *W2 + b2 -> relu  = concept layer -> *W3 + b3 -> relu -> *W4 + b4 -> relu = concept layer -> W3 etc.

ninput = 9
nhidden = 10
nconcept = 5
nconcept_targets = 5 #first 5 are outputs: next center life value, current center life value, live neighbors >= 2, live neighbors >= 3, live_neighbors >= 4

nsamples = 256

eta = 0.01
nepochs = 50


#Compositional
W1 = tf.Variable(tf.random_uniform([nhidden,ninput],0,1)) 
b1 = tf.Variable(tf.random_uniform([nhidden,1],0,1)) 
W2 = tf.Variable(tf.random_uniform([nconcept,nhidden],0,1)) 
b2 = tf.Variable(tf.random_uniform([nconcept,1],0,1)) 
W3 = tf.Variable(tf.random_uniform([nhidden,nconcept],0,1)) 
b3 = tf.Variable(tf.random_uniform([nhidden,1],0,1)) 
W4 = tf.Variable(tf.random_uniform([nconcept,nhidden],0,1)) 
b4 = tf.Variable(tf.random_uniform([nconcept,1],0,1)) 

input_ph = tf.placeholder(tf.float32,shape=[ninput,1])
concept_target = tf.placeholder(tf.float32,shape=[nconcept_targets,1])

def perceive(array): #Maps from input to concepts
    return tf.nn.sigmoid(tf.matmul(W2,tf.nn.sigmoid(tf.matmul(W1,array)+b1))+ b2)

def contemplate_concepts(concept_array): #Maps from concepts to concepts
    return tf.nn.sigmoid(tf.matmul(W4,tf.nn.sigmoid(tf.matmul(W3,concept_array)+b3))+ b4)


perceived_concepts = perceive(input_ph)
contemplation_concepts = contemplate_concepts(perceived_concepts)


perception_error = tf.square(concept_target-perceived_concepts)
perception_output_error = tf.square(concept_target[0,0]-perceived_concepts[0,0]) #Just the error on the final output value
contemplation_error = tf.square(concept_target-contemplation_concepts)
contemplation_output_error = tf.square(concept_target[0,0]-contemplation_concepts[0,0]) #Just the error on the final output value

perception_loss = tf.reduce_sum(perception_error)
contemplation_loss = tf.reduce_sum(contemplation_error)


optimizer = tf.train.AdamOptimizer(eta)
optimizer2 = tf.train.AdamOptimizer(eta)


perception_train = optimizer.minimize(perception_loss)
contemplation_train = optimizer2.minimize(contemplation_loss)


#Standard
standard_W1 = tf.Variable(W1.initialized_value())
standard_b1 = tf.Variable(b1.initialized_value())
standard_W2 = tf.Variable(W2.initialized_value())
standard_b2 = tf.Variable(b2.initialized_value())
standard_W3 = tf.Variable(W3.initialized_value())
standard_b3 = tf.Variable(b3.initialized_value())
standard_W4 = tf.Variable(W4.initialized_value())
standard_b4 = tf.Variable(b4.initialized_value())

def standard_perceive(array): #Maps from input to concepts
    return tf.nn.sigmoid(tf.matmul(standard_W2,tf.nn.sigmoid(tf.matmul(standard_W1,array)+standard_b1))+ standard_b2)

def standard_contemplate_concepts(concept_array): #Maps from concepts to concepts
    return tf.nn.sigmoid(tf.matmul(standard_W4,tf.nn.sigmoid(tf.matmul(standard_W3,concept_array)+standard_b3))+ standard_b4)


standard_perceived_concepts = standard_perceive(input_ph)
standard_contemplation_concepts = standard_contemplate_concepts(standard_perceived_concepts)


standard_perception_output_error = tf.square(concept_target[0,0]-standard_perceived_concepts[0,0]) #Just the error on the final output value
standard_contemplation_output_error = tf.square(concept_target[0,0]-standard_contemplation_concepts[0,0]) #Just the error on the final output value

standard_perception_loss = tf.reduce_sum(standard_perception_output_error)
standard_contemplation_loss = tf.reduce_sum(standard_contemplation_output_error)

standard_optimizer = tf.train.AdamOptimizer(eta)
standard_optimizer2 = tf.train.AdamOptimizer(eta)

standard_perception_train = standard_optimizer.minimize(standard_perception_loss)
standard_contemplation_train = standard_optimizer2.minimize(standard_contemplation_loss)


#data building

def ith_binary_array(i,n=9):
    """Returns the ith (0-indexed) binary array from the enumeration of all 2^n binary array with n elements, in order [0,0,...,0],[1,0,...,0], etc."""
    return numpy.array([(i//2**j) % 2 for j in xrange(n)]) 

def get_concept_target(a):
    """Gets concept target values for an array, i.e. next life value, current life value, number live neighbors""" 
    curr_life = a[4]
    num_live = numpy.sum(a)-curr_life
    nl_g1 = 0
    nl_g2 = 0
    nl_g3 = 0
    if (num_live > 1):
	nl_g1 = 1
    if (num_live > 2):
	nl_g2 = 1
    if (num_live > 3):
	nl_g3 = 1
    if (num_live > 3 or num_live < 2):
	next_life = 0
    elif num_live == 3:
	next_life = 1
    else:
	next_life = curr_life
    return numpy.array([next_life,curr_life,nl_g1,nl_g2,nl_g3])

numbers = numpy.arange(512) 
numpy.random.shuffle(numbers)
training_data = numbers[:nsamples]
testing_data = numbers[nsamples:]

#Initialize
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)



perception_MSE_track = []
contemplation_MSE_track = []
standard_perception_MSE_track = []
standard_contemplation_MSE_track = []

def calculate_MSEs():
    #assess on testing data
    perception_MSE = 0.
    for i in xrange(512-nsamples):
	    perception_MSE += numpy.mean(sess.run(perception_output_error,feed_dict={input_ph: ith_binary_array(testing_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(testing_data[i])).reshape([nconcept_targets,1])}))
    perception_MSE /= 512-nsamples
    contemplation_MSE = 0.
    for i in xrange(512-nsamples):
	    contemplation_MSE += numpy.mean(sess.run(contemplation_output_error,feed_dict={input_ph: ith_binary_array(testing_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(testing_data[i])).reshape([nconcept_targets,1]) }))
    contemplation_MSE /= 512-nsamples
    standard_perception_MSE = 0.
    for i in xrange(512-nsamples):
	    standard_perception_MSE += numpy.mean(sess.run(standard_perception_output_error,feed_dict={input_ph: ith_binary_array(testing_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(testing_data[i])).reshape([nconcept_targets,1])}))
    standard_perception_MSE /= 512-nsamples
    standard_contemplation_MSE = 0.
    for i in xrange(512-nsamples):
	    standard_contemplation_MSE += numpy.mean(sess.run(standard_contemplation_output_error,feed_dict={input_ph: ith_binary_array(testing_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(testing_data[i])).reshape([nconcept_targets,1]) }))
    standard_contemplation_MSE /= 512-nsamples
    return (perception_MSE,contemplation_MSE,standard_perception_MSE,standard_contemplation_MSE)
MSEs = calculate_MSEs()
print("Initial MSEs: ", MSEs)

perception_MSE_track.append(MSEs[0])
contemplation_MSE_track.append(MSEs[1])
standard_perception_MSE_track.append(MSEs[2])
standard_contemplation_MSE_track.append(MSEs[3])


for epoch in xrange(nepochs):
    print "Running epoch ",epoch
    this_order = numpy.array(range(nsamples))
    numpy.random.shuffle(this_order)
    for i in this_order:
	sess.run(perception_train,feed_dict={input_ph: ith_binary_array(training_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(training_data[i])).reshape([nconcept_targets,1])})
	sess.run(contemplation_train,feed_dict={input_ph: ith_binary_array(training_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(training_data[i])).reshape([nconcept_targets,1])})
	sess.run(standard_perception_train,feed_dict={input_ph: ith_binary_array(training_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(training_data[i])).reshape([nconcept_targets,1])})
	sess.run(standard_contemplation_train,feed_dict={input_ph: ith_binary_array(training_data[i]).reshape([ninput,1]),concept_target: get_concept_target(ith_binary_array(training_data[i])).reshape([nconcept_targets,1])})

    #track
    MSEs = calculate_MSEs()
    perception_MSE_track.append(MSEs[0])
    contemplation_MSE_track.append(MSEs[1])
    standard_perception_MSE_track.append(MSEs[2])
    standard_contemplation_MSE_track.append(MSEs[3])


print("Final MSEs:", MSEs)

plot.plot(range(nepochs+1),perception_MSE_track,label='Perception')
plot.plot(range(nepochs+1),contemplation_MSE_track,label='Contemplation')
plot.plot(range(nepochs+1),standard_perception_MSE_track,label='Std. Perc.')
plot.plot(range(nepochs+1),standard_contemplation_MSE_track,label='Std. Cont.')
plot.legend()
plot.xlabel('Epochs')
plot.ylabel('MSE')
plot.show()
