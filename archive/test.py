from torchrl.data import BoundedTensorSpec
from torchrl.modules import SafeModule
from torchrl.envs.libs.gym import GymEnv, GymWrapper
# Import torch 
import torch
import torch.nn as nn
# Import tensordict
from tensordict import TensorDict
from torchrl.envs.utils import step_mdp

env = GymEnv("Pendulum-v1")
torch.manual_seed(0)
env.set_seed(0)
action_spec = env.action_spec
obs_spec = env.observation_space
actor_module = nn.Linear(3, 1)
actor = SafeModule(
    actor_module, spec=action_spec, in_keys=["observation"], out_keys=["action"]
)
critic_module = nn.Linear(3, 1)
critic = SafeModule(critic_module,spec=action_spec, in_keys=["observation"], out_keys=["value"])


print("Obs space of size {}".format(obs_spec))
print("Action space of size {} with minimum {} and maximum {}".format(action_spec.shape, action_spec.space.minimum, action_spec.space.maximum))

max_steps = 1000
tensordict = env.reset()
tensordicts = TensorDict({}, [max_steps]) 
for i in range(max_steps):
    actor(tensordict)
    tensordicts[i] = env.step(tensordict)
    print(tensordict['reward'])
    if tensordict["done"].any():
        break
    
    tensordict = step_mdp(tensordict)  # roughly equivalent to obs = next_obs

print(tensordict)
