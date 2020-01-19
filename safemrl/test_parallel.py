import tensorflow as tf
tf.enable_v2_behavior()

from safemrl.envs import cube_env
from safemrl.utils import metrics
from tf_agents.environments import suite_mujoco, parallel_py_environment, tf_py_environment
import pickle

py_env = parallel_py_environment.ParallelPyEnvironment(
    [lambda: suite_mujoco.load('SafemrlCube-v0', gym_env_wrappers=[cube_env.CubeTaskAgnWrapper])] * 3
)
try:
    print('pickles:', pickle.dumps(py_env._envs[0].gym))
except pickle.PicklingError:
    print('did not pickle')
