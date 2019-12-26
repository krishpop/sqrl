import gin
import tensorflow as tf

from gym.wrappers import monitor
from tf_agents.metrics import tf_py_metric

gin.external_configurable(tf_py_metric.TFPyMetric,
                          module='tf_agents.metrics.tf_py_metric')
gin.external_configurable(monitor.Monitor, module='gym.wrappers.monitor')
gin.external_configurable(tf.keras.layers.Concatenate, module='tf.keras.layers')