import tensorflow as tf
tf.compat.v1.enable_v2_behavior()

from tf_agents.trajectories import trajectory


def eval_fn(before_fail_step, fail_step, safe_step, after_safe_step, get_action):
  time_step, _, _ = trajectory.to_transition(before_fail_step, fail_step)
  fail_pol_ac = get_action(time_step)
  fail_ac = fail_step.action

  time_step, _, _ = trajectory.to_transition(safe_step, after_safe_step)
  init_pol_ac = get_action(time_step)
  init_ac = safe_step.action

  fail_pol_sc_input = (fail_step.observation, fail_pol_ac)
  fail_sc_input = (fail_step.observation, fail_ac)
  init_pol_sc_input = (safe_step.observation, init_pol_ac)
  init_sc_input = (safe_step.observation, init_ac)

  def eval_sc(safety_critic, step):
    p_fail_logits, _ = safety_critic(fail_pol_sc_input, fail_step.step_type,
                                     training=True)
    p_fail_onpol = tf.nn.sigmoid(p_fail_logits)

    p_fail_logits, _ = safety_critic(fail_sc_input, fail_step.step_type,
                                     training=True)
    p_fail = tf.nn.sigmoid(p_fail_logits)

    init_p_fail_logits, _ = safety_critic(init_pol_sc_input,
                                          safe_step.step_type, training=True)
    init_p_fail_onpol = tf.nn.sigmoid(init_p_fail_logits)

    init_p_fail_logits, _ = safety_critic(init_sc_input, safe_step.step_type,
                                          training=True)
    init_p_fail = tf.nn.sigmoid(init_p_fail_logits)

    tf.compat.v2.summary.histogram('p_fail_term_onpol', data=p_fail_onpol,
                                   step=step)
    tf.compat.v2.summary.histogram('p_fail_term', data=p_fail, step=step)
    tf.compat.v2.summary.histogram('p_fail_init_onpol',
                                   data=init_p_fail_onpol, step=step)
    tf.compat.v2.summary.histogram('p_fail_init', data=init_p_fail, step=step)

  return eval_sc