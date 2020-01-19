import collections
import gin
import tensorflow as tf
import tensorflow_probability as tfp

from safemrl.algos import agents
from tf_agents.agents import tf_agent
from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.utils import nest_utils

ds = tfp.distributions

class WcpgInfo(collections.namedtuple(
    'WcpgInfo', ('actor_loss', 'critic_loss', 'mean_loss', 'var_loss'))):
  pass


@gin.configurable
class WcpgAgent(ddpg_agent.DdpgAgent):
  def __init__(self,
               time_step_spec,
               action_spec,
               actor_network,
               critic_network,
               actor_optimizer=None,
               critic_optimizer=None,
               exploration_noise_stddev=0.1,
               target_actor_network=None,
               target_critic_network=None,
               target_update_tau=1.0,
               target_update_period=1,
               dqda_clipping=None,
               td_errors_loss_fn=tf.math.squared_difference,
               gamma=1.0,
               reward_scale_factor=1.0,
               gradient_clipping=None,
               debug_summaries=False,
               summarize_grads_and_vars=False,
               train_step_counter=None,
               name=None):

    """Creates a DDPG Agent.

    Args:
      time_step_spec: A `TimeStep` spec of the expected time_steps.
      action_spec: A nest of BoundedTensorSpec representing the actions.
      actor_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call(observation, step_type[, policy_state])
        and should return (action, new_state).
      critic_network: A tf_agents.network.Network to be used by the agent. The
        network will be called with call((observation, action), step_type[,
        policy_state]) and should return (q_value, new_state).
      actor_optimizer: The optimizer to use for the actor network.
      critic_optimizer: The optimizer to use for the critic network.
      ou_stddev: Standard deviation for the Ornstein-Uhlenbeck (OU) noise added
        in the default collect policy.
      ou_damping: Damping factor for the OU noise added in the default collect
        policy.
      target_actor_network: (Optional.)  A `tf_agents.network.Network` to be
        used as the actor target network during Q learning.  Every
        `target_update_period` train steps, the weights from `actor_network` are
        copied (possibly withsmoothing via `target_update_tau`) to `
        target_q_network`.

        If `target_actor_network` is not provided, it is created by making a
        copy of `actor_network`, which initializes a new network with the same
        structure and its own layers and weights.

        Performing a `Network.copy` does not work when the network instance
        already has trainable parameters (e.g., has already been built, or
        when the network is sharing layers with another).  In these cases, it is
        up to you to build a copy having weights that are not
        shared with the original `actor_network`, so that this can be used as a
        target network.  If you provide a `target_actor_network` that shares any
        weights with `actor_network`, a warning will be logged but no exception
        is thrown.
      target_critic_network: (Optional.) Similar network as target_actor_network
         but for the critic_network. See documentation for target_actor_network.
      target_update_tau: Factor for soft update of the target networks.
      target_update_period: Period for soft update of the target networks.
      dqda_clipping: when computing the actor loss, clips the gradient dqda
        element-wise between [-dqda_clipping, dqda_clipping]. Does not perform
        clipping if dqda_clipping == 0.
      td_errors_loss_fn:  A function for computing the TD errors loss. If None,
        a default value of elementwise huber_loss is used.
      gamma: A discount factor for future rewards.
      reward_scale_factor: Multiplicative scale for the reward.
      gradient_clipping: Norm length to clip gradients.
      debug_summaries: A bool to gather debug summaries.
      summarize_grads_and_vars: If True, gradient and network variable summaries
        will be written during training.
      train_step_counter: An optional counter to increment every time the train
        op is run.  Defaults to the global_step.
      name: The name of this agent. All variables in this module will fall
        under that name. Defaults to the class name.
    """
    tf.Module.__init__(self, name=name)
    self._actor_network = actor_network
    actor_network.create_variables()
    if target_actor_network:
      target_actor_network.create_variables()
    self._target_actor_network = common.maybe_copy_target_network_with_checks(
        self._actor_network, target_actor_network, 'TargetActorNetwork')
    self._critic_network = critic_network
    critic_network.create_variables()
    if target_critic_network:
      target_critic_network.create_variables()
    self._target_critic_network = common.maybe_copy_target_network_with_checks(
        self._critic_network, target_critic_network, 'TargetCriticNetwork')

    self._actor_optimizer = actor_optimizer
    self._critic_optimizer = critic_optimizer

    self._standard_normal = ds.Normal(0, 1)
    self._target_update_tau = target_update_tau
    self._target_update_period = target_update_period
    self._dqda_clipping = dqda_clipping
    self._td_errors_loss_fn = (
        td_errors_loss_fn or common.element_wise_huber_loss)
    self._gamma = gamma
    self._reward_scale_factor = reward_scale_factor
    self._gradient_clipping = gradient_clipping

    self._update_target = self._get_target_updater(
        target_update_tau, target_update_period)

    policy = agents.WcpgPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec,
        actor_network=self._actor_network, clip=True)
    collect_policy = agents.WcpgPolicy(
        time_step_spec=time_step_spec, action_spec=action_spec,
        actor_network=self._actor_network, clip=False)
    collect_policy = agents.GaussianNoisePolicy(
        collect_policy,
        exploration_noise_stddev=exploration_noise_stddev,
        clip=True)

    super(ddpg_agent.DdpgAgent, self).__init__(
        time_step_spec,
        action_spec,
        policy,
        collect_policy,
        train_sequence_length=2 if not self._actor_network.state_spec else None,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)

  def _experience_to_transitions(self, experience):
    boundary_mask = experience.is_boundary()[:, 0]
    experience = nest_utils.fast_map_structure(lambda *x: tf.boolean_mask(*x, boundary_mask), experience)
    time_steps, policy_steps, next_time_steps = trajectory.to_transition(experience)

    actions = policy_steps.action
    if (self.train_sequence_length is not None and
            self.train_sequence_length == 2):
      # Sequence empty time dimension if critic network is stateless.
      time_steps, actions, next_time_steps = tf.nest.map_structure(
        lambda t: tf.squeeze(t, axis=1),
        (time_steps, actions, next_time_steps))
    return time_steps, actions, policy_steps.info.alpha[:,0], next_time_steps

  def _train(self, experience, weights=None):
    time_steps, actions, alphas, next_time_steps = self._experience_to_transitions(experience)

    trainable_critic_variables = self._critic_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_critic_variables, ('No trainable critic variables to '
                                          'optimize.')
      tape.watch(trainable_critic_variables)
      critic_loss, mean_loss, var_loss = self.critic_loss(
          time_steps, actions, alphas, next_time_steps, weights=weights)
    tf.debugging.check_numerics(critic_loss, 'Critic loss is inf or nan.')
    critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
    self._apply_gradients(critic_grads, trainable_critic_variables,
                          self._critic_optimizer)

    trainable_actor_variables = self._actor_network.trainable_variables
    with tf.GradientTape(watch_accessed_variables=False) as tape:
      assert trainable_actor_variables, ('No trainable actor variables to '
                                         'optimize.')
      tape.watch(trainable_actor_variables)
      actor_loss = self.actor_loss(time_steps, alphas, weights=weights)
    tf.debugging.check_numerics(actor_loss, 'Actor loss is inf or nan.')
    actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
    self._apply_gradients(actor_grads, trainable_actor_variables,
                          self._actor_optimizer)

    self.train_step_counter.assign_add(1)
    self._update_target()

    # TODO(b/124382360): Compute per element TD loss and return in loss_info.
    total_loss = actor_loss + critic_loss
    return tf_agent.LossInfo(total_loss,
                             WcpgInfo(actor_loss, critic_loss, mean_loss, var_loss))

  def critic_loss(self, time_steps, actions, alphas, next_time_steps, weights=None):
    """Computes the critic loss for DDPG training.

    Args:
      time_steps: A batch of timesteps.
      actions: A batch of actions.
      next_time_steps: A batch of next timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
    Returns:
      critic_loss: A scalar critic loss.
    """
    with tf.name_scope('critic_loss'):
      target_actions, _ = self._target_actor_network(
          (next_time_steps.observation, alphas), next_time_steps.step_type)
      next_target_critic_net_input = (next_time_steps.observation, target_actions, alphas)
      next_target_Z, _ = self._target_critic_network(
          next_target_critic_net_input, next_time_steps.step_type)
      next_target_means = tf.reshape(next_target_Z.loc, [-1])
      next_target_vars = tf.reshape(next_target_Z.scale, [-1])
      target_critic_net_input = (time_steps.observation, target_actions, alphas)
      target_Z, _ = self._target_critic_network(
        target_critic_net_input, next_time_steps.step_type)
      target_means = tf.reshape(target_Z.loc, [-1])
      if len(next_target_means.shape) != 1:
        raise ValueError('Q-network should output a tensor of shape (batch,) '
                         'but shape {} was returned.'.format(
                             next_target_means.shape.as_list()))
      if len(target_means.shape) != 1:
        raise ValueError('Q-network should output a tensor of shape (batch,) '
                         'but shape {} was returned.'.format(
                             target_means.shape.as_list()))

      td_mean_target = tf.stop_gradient(
          self._reward_scale_factor * next_time_steps.reward +
          self._gamma * next_time_steps.discount * next_target_means)

      # Refer to Eq. 8 in WCPG
      td_var_target = tf.stop_gradient(
        (self._reward_scale_factor * next_time_steps.reward) ** 2 +
         2 * self._gamma * next_time_steps.discount * next_time_steps.reward * next_target_means +
         next_time_steps.discount * self._gamma ** 2 * next_target_vars + self._gamma ** 2 *
         next_target_means - next_time_steps.discount * target_means ** 2)

      critic_net_input = (time_steps.observation, actions, alphas)
      Z, _ = self._critic_network(critic_net_input,
                                  time_steps.step_type)
      q_means, q_vars = Z.loc, Z.scale
      mean_td_error = self._td_errors_loss_fn(td_mean_target, q_means)
      var_td_error = tf.sqrt(self._td_errors_loss_fn(td_var_target, q_vars))
      critic_loss = mean_td_error + var_td_error

      if nest_utils.is_batched_nested_tensors(
          time_steps, self.time_step_spec, num_outer_dims=2):
        # Do a sum over the time dimension.
        critic_loss = tf.reduce_sum(critic_loss, axis=1)
      if weights is not None:
        critic_loss *= weights
      critic_loss = tf.reduce_mean(critic_loss)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='critic_loss', data=critic_loss, step=self.train_step_counter)

      if self._debug_summaries:
        mean_td_errors = td_mean_target - q_means
        var_td_errors = td_var_target - q_vars
        common.generate_tensor_summaries('mean_td_errors', mean_td_errors,
                                         self.train_step_counter)
        common.generate_tensor_summaries('var_td_errors', var_td_errors,
                                         self.train_step_counter)
        common.generate_tensor_summaries('td_mean_targets', td_mean_target,
                                         self.train_step_counter)
        common.generate_tensor_summaries('td_var_targets', td_var_target,
                                         self.train_step_counter)
        common.generate_tensor_summaries('q_mean', q_means,
                                         self.train_step_counter)
        common.generate_tensor_summaries('q_var', q_vars,
                                         self.train_step_counter)

      return critic_loss, mean_td_error, var_td_error

  def _compute_cvar(self, q_means, q_vars, alpha):
    return (q_means - self._standard_normal.prob(alpha)/self._standard_normal.cdf(alpha) *
            tf.sqrt(q_vars))

  def actor_loss(self, time_steps, alphas, weights=None):
    """Computes the actor_loss for DDPG training.

    Args:
      time_steps: A batch of timesteps.
      weights: Optional scalar or element-wise (per-batch-entry) importance
        weights.
      # TODO(b/124383618): Add an action norm regularizer.
    Returns:
      actor_loss: A scalar actor loss.
    """
    with tf.name_scope('actor_loss'):
      actions, _ = self._actor_network((time_steps.observation, alphas),
                                       time_steps.step_type)
      with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(actions)
        q, _ = self._critic_network((time_steps.observation, actions, alphas),
                                                     time_steps.step_type)
        q_means, q_vars = q.loc, q.scale
        actions = tf.nest.flatten(actions)

      cvar = self._compute_cvar(q_means, q_vars, alphas)
      actor_loss = tf.reduce_mean(cvar)

      with tf.name_scope('Losses/'):
        tf.compat.v2.summary.scalar(
            name='actor_loss', data=actor_loss, step=self.train_step_counter)

    return actor_loss