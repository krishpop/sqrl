from tf_agents.metrics import tf_py_metric
from tf_agents.metrics import py_metrics

import gin

gin.external_configurable(tf_py_metric.TFPyMetric,
                          module='tf_agents.metrics.tf_py_metric')
