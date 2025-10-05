import gymnasium as gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
import pygame
import sys

# Create environment with Gymnasium
env = gym_super_mario_bros.make('SuperMarioBros-v0', render_mode="human", apply_api_compatibility=True)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

# Initialize pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("Super Mario - Keyboard Control")

# Map keys to actions
KEY_TO_ACTION = {
    pygame.K_0: 0,      # NOOP
    pygame.K_RIGHT: 1,  # Move right
    pygame.K_LEFT: 2,   # Move left
    pygame.K_z: 3,      # Jump right
    pygame.K_x: 4,      # Jump left
    pygame.K_UP: 5,     # Jump
    pygame.K_DOWN: 6,   # Duck
}

obs, info = env.reset()
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
    action = 0  # Default NOOP

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    keys = pygame.key.get_pressed()
    for key, mapped_action in KEY_TO_ACTION.items():
        if keys[key]:
            action = mapped_action
            break

    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    clock.tick(60)

    if done:
        obs, info = env.reset()

env.close()
pygame.quit()
sys.exit()
