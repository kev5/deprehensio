from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import numpy as np
import tensorflow as tf
from model import select_model, get_checkpoint
from utils import ImageCoder, make_batch

def get_files(path):
    # return the path of all the files in the dir
    file_lists = []

    files = os.listdir(path)
    for f in files:
        if os.path.isfile(path + '/' + f):
            if (f[0] != '.') & (f[-1] == 'g'):
                file_lists.append(path + '/' + f)
    return file_lists


AGE_MODEL_PATH = '/Users/apple/Desktop/try/age_model'
GENDER_MODEL_PATH = './gender_model'

RESIZE_FINAL = 227
GENDER_LIST = ['MALE', 'FEMALE']
AGE_LIST = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model directory (where training data lives)')

tf.app.flags.DEFINE_string('class_type', 'gender',
                           'Classification type (age|gender)')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('filename', '',
                           'File (Image) or File list (Text/No header TSV) to process')

tf.app.flags.DEFINE_string('target', '',
                           'CSV file containing the filename processed along with best guess and score')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                           'Checkpoint basename')

tf.app.flags.DEFINE_string('model_type', 'inception model',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step', '', 'Within the model directory, a requested step to restore e.g., 9000')

tf.app.flags.DEFINE_boolean('single_look', False, 'single look at the image or multiple crops')

tf.app.flags.DEFINE_string('face_detection_model', '', 'Do frontal face detection with model specified')

tf.app.flags.DEFINE_string('face_detection_type', 'cascade', 'Face detection model type (yolo_tiny|cascade)')

FLAGS = tf.app.flags.FLAGS


def classify(sess, label_list, softmax_output, coder, images, image_file):
    print('Running file %s' % image_file)
    image_batch = make_batch(image_file, coder, not FLAGS.single_look)
    batch_results = sess.run(softmax_output, feed_dict={images: image_batch.eval()})
    output = batch_results[0]
    batch_sz = batch_results.shape[0]
    for i in range(1, batch_sz):
        output = output + batch_results[i]

    output /= batch_sz
    best = np.argmax(output)
    best_choice = (label_list[best], output[best])
    print('Guess @ 1 %s, prob = %.2f' % best_choice)

    nlabels = len(label_list)
    if nlabels > 2:
        output[best] = 0
        second_best = np.argmax(output)

        print('Guess @ 2 %s, prob = %.2f' % (label_list[second_best], output[second_best]))
    return best_choice




def guessGender(path):  # pylint: disable=unused-argument
# 检测文件夹中所有照片的性别

    with tf.Session() as sess:

        # tf.reset_default_graph()
        label_list = GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % FLAGS.device_id)
        model_fn = select_model(FLAGS.model_type)

        with tf.device(FLAGS.device_id):

            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()

            requested_step = FLAGS.requested_step if FLAGS.requested_step else None

            checkpoint_path = '%s' % (GENDER_MODEL_PATH)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)
            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)

            softmax_output = tf.nn.softmax(logits)

            coder = ImageCoder()
            files = get_files(path)
            gender_dict = {}

            try:
                for f in files:
                    best_choice = classify(sess, label_list, softmax_output, coder, images, f)
                    # print(best_choice)
                    gender_dict[f[len(path) + 1:]] = best_choice
                    return(best_choice)


            except Exception as e:
                print(e)
                print('Failed to run image %s ' % file)

if __name__ == '__main__':
    tf.app.run()
