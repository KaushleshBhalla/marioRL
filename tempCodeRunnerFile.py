import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
import sys

# Set up environment with latest version v3 for Gymnasium compatibility
env = gym_super_mario_bros.make('SuperMarioBros-v3', render_mode='human')
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Initialize pygame just for capturing keyboard input
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Super Mario - Keyboard Control")

# Mapping of key presses to actions in SIMPLE_MOVEMENT
KEY_TO_ACTION = {
    pygame.K_0: 0,  # NOOP
    pygame.K_RIGHT: 1,  # right
    pygame.K_LEFT: 2,   # left
    pygame.K_z: 3,      # jump right
    pygame.K_x: 4,      # jump left
    pygame.K_UP: 5,     # jump
    pygame.K_DOWN: 6,   # duck
}

# Reset environment
state, info = env.reset()
clock = pygame.time.Clock()

print("Controls:")
print("➡️ Right Arrow = Move Right")
print("⬅️ Left Arrow  = Move Left")
print("Z = Jump Right")
print("X = Jump Left")
print("↑ Up Arrow    = Jump")
print("↓ Down Arrow  = Duck")
print("ESC to quit")

running = True
while running:
    action = 0  # default to NOOP

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
            break
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
                break

    keys = pygame.key.get_pressed()
    for key, mapped_action in KEY_TO_ACTION.items():
        if keys[key]:
            action = mapped_action
            break

    state, reward, done, truncated, info = env.step(action)
    # The env automatically renders because of render_mode='human'

    clock.tick(60)

    if done or truncated:
        state, info = env.reset()

# Cleanup
env.close()
pygame.quit()
sys.exit()
