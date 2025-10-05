from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
env = gym_super_mario_bros.make("SuperMarioBros-v0", render_mode='human', apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
print(SIMPLE_MOVEMENT)
for step in range(50):
    if done:
        state = env.reset()
    state,reward,done,noise,info = env.step(env.action_space.sample())
    print(info,reward)
    env.render()

# env.close()