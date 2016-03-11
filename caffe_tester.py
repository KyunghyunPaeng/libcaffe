"""
Wrapper for caffe library
Written by Paeng
"""
import os
import re
import sys
import caffe
import numpy as np
import skimage

class Tester(object):
    def __init__(self, deploy_prototxt, caffemodel, bgr_order_mean, on_gpu=None, vis_mode=False):
        if on_gpu is not None:
            caffe.set_device(on_gpu)
            caffe.set_mode_gpu()
        if vis_mode is True:
            model = caffe.io.caffe_pb2.NetParameter()
            from google.protobuf import text_format
            text_format.Merge(open(deploy_prototxt).read(), model)
            model.force_backward = True
            open('tmp.prototxt', 'w').write(str(model))
            deploy_prototxt = 'tmp.prototxt'
        self._net = caffe.Net(deploy_prototxt, caffemodel, caffe.TEST)
        self._name = os.path.splitext(os.path.basename(caffemodel))[0]
        self._net_inputs = self._net.inputs
        self._net_outputs = self._net.outputs
        self._net_layers = self._parse_deploy(deploy_prototxt)
        self._mean = np.array(bgr_order_mean).astype(np.float32)
        self._scale = 255.0 # default scale

    @property
    def mean(self):
        return self._mean
    @property
    def layers(self):
        return self._net_layers
    @property
    def inputs(self):
        return self._net_inputs
    @property
    def outputs(self):
        return self._net_outputs
    @property
    def input_shape(self):
        shapes = []
        for i, input_name in enumerate(self._net_inputs):
            shapes.append(self._net.blobs[input_name].data.shape)
        return shapes
 
    def target_shape(self, target_layer):
        return self._net.blobs[target_layer].data.shape

    def activations(self, layer_name):
        return self._net.blobs[layer_name].data.copy()
   
    def run_forward(self, img_or_path, resize_shape=None, intensity_scale=None):
        """Read a image using skimage library (RGB order!)
           Must convert BGR order!!
           resize_shape = [ height, width ]"""
        if intensity_scale is not None:
            self._scale = intensity_scale
        for i, input_name in enumerate(self._net_inputs):
            shape = self._net.blobs[input_name].data.shape
            if resize_shape is not None:
                shape[2] = resize_shape[0]
                shape[3] = resize_shape[1]
            in_ = self._prepare_batch(img_or_path, shape)
            if shape[2:]==in_.shape[2:] is False:
                self._net.blobs[input_name].reshape(*(in_.shape)) # reshape
            self._net.blobs[input_name].data[...] = in_
        out = self._net.forward()
        self._output = out[self._net_layers[-1]].copy()
        return self._output

    def _prepare_batch(self, img, shape):
        image_dim = shape[2:]
        channel_dim = shape[1]
        batch_size = shape[0]
        if type(img) is str:
            im = self._load_image(img, channel_dim)
        else:
            im = img
            if im.dtype is not np.dtype('float32'):
                im = skimage.img_as_float(im).astype(np.float32)
        # warping (default mode)
        if im.shape[:2] != image_dim:
            im = skimage.transform.resize(im, (image_dim[0], image_dim[1]))
        im = im*self._scale
        if channel_dim == 3:
            im = im[:, :, ::-1]
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
        im = im.transpose((2, 0, 1))
        im = im - self._mean[:, np.newaxis, np.newaxis]
        batch = np.zeros( (batch_size, channel_dim, image_dim[0], image_dim[1]), dtype=np.float32 ) 
        batch[0] = im
        return batch

    def _load_image(self, image_path, channel):
        if channel == 1:
            im = skimage.img_as_float(skimage.io.imread(image_path, as_grey=True)).astype(np.float32)
        else:
            im = skimage.img_as_float(skimage.io.imread(image_path, as_grey=False)).astype(np.float32)
        return im
    
    def _convert_batch_to_img(self, batch, clip=False, mean=True):
        if batch.shape[0] != 1:
            assert False, 'Not supported multiple images'
        else:
            img = batch[0]
            if mean:
                img = img + self._mean[:, np.newaxis, np.newaxis]
            img = img.astype(np.float32)
            img = img.transpose((1,2,0))
            if img.shape[2] == 3:
                img = img[:,:,::-1]
            elif img.shape[2] == 1:
                img = img.squeeze()
            if clip:
                img = np.clip(img, 0.0, self._scale)
                img = img/self._scale
            else:
                img = img/self._scale
                img = img-img.min()
                img = img/img.max()
        return img

    def _parse_deploy(self, deploy):
        layer_list = []
        with open(deploy, 'r') as f:
            deploy_str = f.read()
        layer_pattern = 'layer[\\s\\S]+?\\n}\\n'
        layers = re.findall(layer_pattern, deploy_str)
        for index, layer in enumerate(layers):
            params = [i.strip() for i in layer.split('\n')]
            for param in params :
                if 'name' in param:
                    current_layer_name = param.split()[-1][1:-1]
                    layer_list.append(current_layer_name)
        return layer_list
	
