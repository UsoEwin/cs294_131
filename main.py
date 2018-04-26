import tensorflow as tf
from  srcnn import SRCNN
flags = tf.app.flags
FLAGS = flags.FLAGS

#flags for tf
flags.DEFINE_integer("c_dim", 3, "Dimension of color channels")
flags.DEFINE_integer("epoch", 8000, "Number of epoch")
flags.DEFINE_integer("image_size", 33, "The size of image input")
flags.DEFINE_integer("label_size", 21, "The size of image output")
#for testing use false
flags.DEFINE_boolean("mode", False, "if train")
flags.DEFINE_integer("stride", 19, "the size of stride")
#saving the check point for training data
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory")
flags.DEFINE_float("learning_rate", 1e-4 , "The learning rate")
flags.DEFINE_integer("batch_size", 128, "the size of batch")
flags.DEFINE_integer("scale", 1, "the size of scale factor for preprocessing input image")# tried 1,2,3
flags.DEFINE_string("result_dir", "result", "Name of result directory")
#to test the different images
flags.DEFINE_string("test_img", "", "test_img")



def main(_): 
    # run the tf
    with tf.Session() as sess:
        srcnn = SRCNN(sess,
                      image_size = FLAGS.image_size,
                      label_size = FLAGS.label_size,
                      c_dim = FLAGS.c_dim)
        srcnn.train(FLAGS)

    # unlike keras, we have to count num of parameters by ourselves
    total_parameters = 0
    for variable in tf.trainable_variables():
    	shape = variable.get_shape()
    	variable_parameters = 1
    	for dim in shape:
    		variable_parameters *= dim.value
    	total_parameters += variable_parameters
    print(total_parameters)

if __name__=='__main__':
    tf.app.run()
