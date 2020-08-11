import tensorflow as tf
tf.compat.v1.enable_v2_behavior()  # ensures v2 enabled, >=1.15 compatibility

from tf_agents.utils import common

def get_target_updater(sc_net, target_sc_net,
                       tau=0.005, period=1.,
                       name='update_target_sc_offline'):
  with tf.name_scope(name):
    def update():
      """Update target network."""
      critic_update = common.soft_variables_update(
        sc_net.variables,
        target_sc_net.variables, tau)
      return critic_update

    return common.Periodically(update, period, 'target_update')


def eval_safety(safety_critic, get_action, time_steps, use_sigmoid=True):
  obs = time_steps.observation
  ac = get_action(time_steps)
  sc_input = (obs, ac)
  q_val, _ = safety_critic(sc_input, time_steps.step_type, training=True)
  if use_sigmoid:
    q_safe = tf.nn.sigmoid(q_val)
  else:
    q_safe = q_val
  return q_safe
