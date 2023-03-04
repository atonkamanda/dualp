"""# Test the environment
env = Env('cartpole-balance', symbolic=False, seed=0, max_episode_length=1000, action_repeat=1, bit_depth=8)
obs = env.reset()
for _ in range(1000):
  print(obs.shape)
  env.render()
  env.step(env.sample_random_action())
env.close()"""

"""# Record a video
env = Env('cartpole-balance', symbolic=False, seed=0, max_episode_length=1000, action_repeat=1, bit_depth=8)
obs = env.reset()
frames = []
for _ in range(1000):
  frames.append(env._env.physics.render(camera_id=0))
  env.step(env.sample_random_action())
env.close()
import imageio
imageio.mimsave('cartpole_balance.gif', frames, fps=60)"""