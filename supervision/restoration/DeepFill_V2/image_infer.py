import warnings
warnings.filterwarnings('ignore')
import os.path

import cv2
import numpy as np
import tensorflow._api.v2.compat.v1 as tf
tf.logging.set_verbosity(tf.logging.ERROR)
tf.disable_v2_behavior()
import neuralgym as ng

from supervision.restoration.DeepFill_V2.inpaint_model import InpaintCAModel

class DeepFillV2ImageInfer(object):
    def __init__(self,
                 checkpoint_dir: str = 'weights/release_celeba_hq_256_deepfill_v2',
                 config_path: str = 'inpaint.yml',
                 batch_size: int = 1,
                 image_height: int = 256,
                 image_width: int = 256,
                 ):
        self.checkpoint_dir = checkpoint_dir
        self.config_path = config_path
        self.FLAGS = ng.Config(config_path)

        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width

        sess_config = tf.ConfigProto(log_device_placement=False)
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config)

        self.model = None
        self.model = self._load_model()

    def _load_model(self, ):
        model = InpaintCAModel()

        self.input_image_ph = tf.placeholder(
            tf.float32, shape=(1, self.image_height, self.image_width * 2, 3))  # TF version only support for bs=1
        output = model.build_server_graph(self.FLAGS, self.input_image_ph)
        output = (output + 1.) * 127.5
        output = tf.reverse(output, [-1])
        output = tf.saturate_cast(output, tf.uint8)
        self.output = output

        vars_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        assign_ops = []
        for var in vars_list:
            vname = var.name
            from_name = vname
            var_value = tf.train.load_variable(self.checkpoint_dir, from_name)
            assign_ops.append(tf.assign(var, var_value))
        self.sess.run(assign_ops)
        print('DeepFillV2 model loaded.')

        return model

    def release(self, ):
        self.sess.close()
        print('DeepFillV2 TensorFlow session closed.')

    def infer_single(self,
                     dropped_img: np.ndarray,
                     drop_mask: np.ndarray,
                     save_folder: str,
                     save_name: str
                     ):
        assert dropped_img.shape == drop_mask.shape
        assert dropped_img.ndim == 3, 'single image input should be (H,W,BGR)'
        assert dropped_img.shape[0] == self.image_height
        assert dropped_img.shape[1] == self.image_width

        h, w, _ = dropped_img.shape
        grid = 8
        image = dropped_img[:h // grid * grid, :w // grid * grid, :]
        mask = drop_mask[:h // grid * grid, :w // grid * grid, :]
        image = np.expand_dims(image, 0)
        mask = np.expand_dims(mask, 0)
        input_image = np.concatenate([image, mask], axis=2)

        # use loaded pretrained model
        result = self.sess.run(self.output, feed_dict={self.input_image_ph: input_image})

        if save_folder != '' and save_name != '':
            save_path = os.path.join(save_folder, save_name)
            cv2.imwrite(save_path, result[0][:, :, ::-1])  # RGB to BGR

        return result

    def _infer_batch(self,
                    input_image: np.ndarray,
                    ):
        """

        :param input_image: (1,H,W,BGR), in [0,255]
        :return: (1,H,W,BGR), in [0,255]
        """
        assert input_image.shape[0] == 1, 'TensorFlow version only supports for bs=1!'

        result = self.sess.run(self.output, feed_dict={self.input_image_ph: input_image})

        return result[:, :, :, ::-1]  # RGB to BGR

    def infer_batch(self,
                    dropped_img: np.ndarray,
                    drop_mask: np.ndarray,
                    save_folder: str,
                    save_name: str
                    ):
        """

        :param dropped_img: (N,H,W,BGR), in [0,255]
        :param drop_mask: (N,H,W,BGR), in [0,255]
        :param save_folder: not used
        :param save_name: not used
        :return: (N,H,W,BGR), in [0,255]
        """
        assert dropped_img.shape == drop_mask.shape
        assert dropped_img.ndim == 4, 'batch image input should be (N,H,W,BGR)'
        assert dropped_img.shape[1] == self.image_height
        assert dropped_img.shape[2] == self.image_width

        b, h, w, _ = dropped_img.shape
        grid = 8
        image = dropped_img[:, :h // grid * grid, :w // grid * grid, :]
        mask = drop_mask[:, :h // grid * grid, :w // grid * grid, :]
        input_image = np.concatenate([image, mask], axis=2)

        ''' Use loaded pretrained model '''
        batch_res = np.zeros_like(dropped_img)
        for b_idx in range(b):
            res = self._infer_batch(input_image[b_idx, None])  # (N,H,W,BGR) to (1,H,W,BGR) to (1,H,W,BGR)
            batch_res[b_idx] = res[0]

        return batch_res


if __name__ == '__main__':
    dropped_img = cv2.imread('examples/places2/case1_input.png')
    drop_mask = cv2.imread('examples/places2/case1_mask.png')
    # dropped_img = cv2.resize(dropped_img, (256, 256))
    # drop_mask = cv2.resize(drop_mask, (256, 256))

    runner = DeepFillV2ImageInfer(checkpoint_dir='weights/release_places2_256_deepfill_v2',
                                  image_height=512,
                                  image_width=680,
                                  batch_size=1)

    ''' 1. Test for single input '''
    res = runner.infer_single(dropped_img,
                              drop_mask,
                              save_folder='examples/',
                              save_name='output.png',
                              )
    print('single infer result shape:', res.shape)

    ''' 2. Test for batch input '''
    dropped_img = np.expand_dims(dropped_img, 0)
    drop_mask = np.expand_dims(drop_mask, 0)
    res = runner.infer_batch(dropped_img,
                             drop_mask,
                             save_folder='examples/',
                             save_name='output.png',
                             )
    print('batch infer result shape:', res.shape)

    runner.release()
