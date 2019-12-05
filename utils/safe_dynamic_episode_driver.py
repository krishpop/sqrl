import gin
import tensorflow as tf

from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.utils import nest_utils
from tf_agents.drivers import dynamic_episode_driver

class SafeDynamicEpisodeDriver(dynamic_episode_driver.DynamicEpisodeDriver):
  def __init__(self,
               env,
               policy,
               temp_rb=None,
               final_rb=None,
               observers=None,
               transition_observers=None,
               num_episodes=1,
               ep_history_unsafe=2,
               unsafe_label='constant'):
    """
    Returns dynamic episode driver with relabeling safety condition on
    ep_history_unsafe states from terminal state.
    Args:
      env:
      policy:
      temp_rb: EpisodicReplayBuffer temporarily storing trajectories as they are being finished
      final_rb: ReplayBuffer that stores completed trajectories
      observers:
      transition_observers:
      num_episodes:
      ep_history_unsafe: Number of states from the end of the episode labeled as "unsafe"
      unsafe_label: Function scaling to generate unsafe label. Options are ['constant', 'linear', 'exp'].
    """
    super(SafeDynamicEpisodeDriver, self).__init__(env, policy, observers,
                                                   transition_observers,
                                                   num_episodes)
    self._ep_history_unsafe = ep_history_unsafe
    self._unsafe_label = unsafe_label
    self._temp_rb = temp_rb
    self._final_rb = final_rb

  def _loop_condition_fn(self, num_episodes):
    """Returns a function with the condition needed for tf.while_loop."""

    def loop_cond(counter, *_):
      """Determines when to stop the loop, based on episode counter.

      Args:
        counter: Episode counters per batch index. Shape [batch_size] when
          batch_size > 1, else shape [].

      Returns:
        tf.bool tensor, shape (), indicating whether while loop should continue.
      """
      return tf.reduce_any(tf.less(counter, 1))

    return loop_cond


  def _loop_body_fn(self):
    """Returns a function with the driver's loop body ops."""

    @tf.function
    def loop_body(counter, time_step, policy_state):
      """Runs a step in environment.

      While loop will call multiple times.

      Args:
        counter: Episode counters per batch index. Shape [batch_size].
        time_step: TimeStep tuple with elements shape [batch_size, ...].
        policy_state: Poicy state tensor shape [batch_size, policy_state_dim].
          Pass empty tuple for non-recurrent policies.

      Returns:
        loop_vars for next iteration of tf.while_loop.
      """
      action_step = self.policy.action(time_step, policy_state)

      # TODO(b/134487572): TF2 while_loop seems to either ignore
      # parallel_iterations or doesn't properly propagate control dependencies
      # from one step to the next. Without this dep, self.env.step() is called
      with tf.control_dependencies(tf.nest.flatten([time_step])):
      # in parallel.
        next_time_step = self.env.step(action_step.action)

      policy_state = action_step.state

      if self._is_bandit_env:
        # For Bandits we create episodes of length 1.
        # Since the `next_time_step` is always of type LAST we need to replace
        # the step type of the current `time_step` to FIRST.
        batch_size = tf.shape(input=time_step.discount)
        time_step = time_step._replace(
          step_type=tf.fill(batch_size, ts.StepType.FIRST))

      traj = trajectory.from_transition(time_step, action_step, next_time_step)

      observer_ops = [observer(traj) for observer in self._observers]
      transition_observer_ops = [
        observer((time_step, action_step, next_time_step))
        for observer in self._transition_observers
      ]
      with tf.control_dependencies(
              [tf.group(observer_ops + transition_observer_ops)]):
        time_step, next_time_step, policy_state = tf.nest.map_structure(
          tf.identity, (time_step, next_time_step, policy_state))

      # While loop counter is only incremented for episode reset episodes.
      # For Bandits, this is every trajectory, for MDPs, this is at boundaries.
      if self._is_bandit_env:
        counter += tf.ones(batch_size, dtype=tf.int32)
      else:
        counter += tf.cast(traj.is_boundary(), dtype=tf.int32)

      if not tf.reduce_any(tf.less(counter, 1)):
        # all episodes have finished:
        for ep_id in range(self._num_episodes):
          episode = self._temp_rb._get_episode(ep_id)
          if episode.observation['task_agn_rew'][-1] == 1:
            rew_type = episode.observation['task_agn_rew'].dtype
            ep_len = episode.observation['task_agn_rew'].shape[0]
            start = max(-self._ep_history_unsafe, -ep_len)
            if self._unsafe_label == 'constant':
              discount = tf.ones((-start,), dtype=rew_type)
            elif self._unsafe_label == 'exp':
              discount = 0.99 ** tf.reverse(tf.range(-start, dtype=rew_type), axis=[0])
            elif self._unsafe_label == 'linear':
              discount = (tf.range(-start, dtype=rew_type) + 1) / -start
            discount = tf.pad(discount, [[ep_len + start, 0]])
            obs = episode.observation
            obs['task_agn_rew'] = discount
            episode._replace(observation=obs)
          trajs = nest_utils.unstack_nested_tensors(episode, self._final_rb.data_spec)
          for traj in trajs:
            self._final_rb.add_batch(traj)
        self._temp_rb.clear()


      return [counter, next_time_step, policy_state]

    return loop_body