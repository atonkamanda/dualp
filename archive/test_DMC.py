from torchrl.envs.libs.dm_control import DMControlEnv

env = DMControlEnv("acrobot", "swingup",from_pixels=True)

def policy(tensordict):
    tensordict.set("action", env.action_spec.rand())
    return tensordict

tensordict = env.reset()
tensordict_rollout = env.rollout(max_steps=100, policy=policy)
env.close()

print(tensordict_rollout)