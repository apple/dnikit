#
# Copyright 2020 Apple Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import typing as t
import tensorflow as tf


def conv_bn_relu_layer(x: tf.keras.layers.Layer,
                       filter_shape: t.Sequence[int],
                       name: str,
                       stride: int = 1,
                       padding: str = 'same',
                       activation: str = 'relu',
                       bn_params: t.Optional[t.Mapping[str, t.Any]] = None
                       ) -> tf.keras.layers.Layer:
    """
    A helper function to conv, batch normalize and relu the input tensor sequentially
    :param filter_shape: list. [filter_height, filter_width, filter_number]
    :return: 4D tensor. Y = Relu(batch_normalize(conv(X)))
    """

    assert len(filter_shape) == 3
    out_channel = filter_shape[-1]

    if bn_params is not None:
        bn_axis = 3 if tf.keras.backend.image_data_format() == 'channels_last' else 1

    x = tf.keras.layers.Conv2D(
        filters=out_channel,
        kernel_size=filter_shape[0:2],
        strides=(stride, stride),
        name=name + '_conv',
        padding=padding
    )(x)

    if bn_params is not None:
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name=name + '_bn', **bn_params)(x)

    if activation:
        x = tf.keras.layers.Activation(activation)(x)

    return x


def get_default_config() -> t.MutableMapping[str, t.Any]:
    return {
        'name': 'keras_simplecnn',

        'network_input': [32, 32, 3],
        'data_format': 'channels_last',

        'epochs': 50,
        'batch_size': 512,
        'lr_step_epochs': 30,
        'lr_step_factor': 0.1,

        'val_epochs': 1,
        'test_epochs': 1,

        'optimizer': 'SGD',
        'SGD_params': {'lr': 0.1,
                       'decay': 0.0,
                       'momentum': 0.9,
                       'nesterov': True
                       },
        'Adam_params': {'lr': 0.001,
                        'beta_1': 0.9,
                        'beta_2': 0.999,
                        'decay': 0.0
                        },

        'drop_out': 0.5,
        'batch_norm': {'momentum': 0.99,
                       'epsilon': 0.001, },
        # 'augment': {
        #     'random_crop': {'size': 32, 'pad': 4},
        #     'horizontal_flip': True,
        # },
        'num_classes': None,
        'dataset': None,
    }


def get_model(config_override: t.Mapping[str, t.Any] = {}) -> tf.keras.models.Model:
    """
    Create SimpleCNN model with the default configuration overridden with keys specified in `config`
    """
    config = get_default_config()
    config.update(config_override)

    classes = config['num_classes']
    input_shape = config['network_input']
    bn_params = config['batch_norm'] if 'batch_norm' in config else None

    # input node
    img_input = tf.keras.layers.Input(shape=input_shape)

    x = conv_bn_relu_layer(
        img_input, filter_shape=[3, 3, 96], name='conv0', bn_params=bn_params)

    x = conv_bn_relu_layer(
        x, filter_shape=[3, 3, 96], name='conv1', bn_params=bn_params)

    x = conv_bn_relu_layer(
        x, filter_shape=[3, 3, 96], stride=2, name='conv2', bn_params=bn_params)

    x = tf.keras.layers.Dropout(config['drop_out'])(x)

    x = conv_bn_relu_layer(
        x, filter_shape=[3, 3, 192], name='conv3', bn_params=bn_params)

    x = conv_bn_relu_layer(
        x, filter_shape=[3, 3, 192], name='conv4', bn_params=bn_params)

    x = conv_bn_relu_layer(
        x, filter_shape=[3, 3, 192], stride=2, name='conv5', bn_params=bn_params)

    x = tf.keras.layers.Dropout(config['drop_out'])(x)

    x = conv_bn_relu_layer(
        x, filter_shape=[3, 3, 192], padding='valid', name='conv6', bn_params=bn_params)

    x = conv_bn_relu_layer(
        x, filter_shape=[1, 1, 192], name='conv7', bn_params=bn_params)

    x = conv_bn_relu_layer(x, filter_shape=[
                           1, 1, classes], activation='', name='conv8', bn_params=bn_params)

    x = tf.keras.layers.GlobalAveragePooling2D(data_format=config['data_format'])(x)

    x = tf.keras.layers.Activation('softmax')(x)

    # Create model.
    inputs = img_input
    model = tf.keras.models.Model(inputs, x, name='sample_model')
    return model
