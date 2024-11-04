# SPDX-FileCopyrightText: Copyright (c) 2022 Guillaume Bellegarda. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2022 EPFL, Guillaume Bellegarda

import os, sys
import gym
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
from sys import platform
# may be helpful depending on your system
# if platform =="darwin": # mac
#   import PyQt5
#   matplotlib.use("Qt5Agg")
# else: # linux
#   matplotlib.use('TkAgg')

# stable-baselines3
from stable_baselines3.common.monitor import load_results 
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3 import PPO, SAC
# from stable_baselines3.common.cmd_util import make_vec_env
from stable_baselines3.common.env_util import make_vec_env # fix for newer versions of stable-baselines3

from env.quadruped_gym_env import QuadrupedGymEnv
# utils
from utils.utils import plot_results
from utils.file_utils import get_latest_model, load_all_results


LEARNING_ALG = "PPO"
interm_dir = "./logs/intermediate_models/"
# path to saved models, i.e. interm_dir + '121321105810'
log_dir = interm_dir + '011024100331'

# initialize env configs (render at test time)
# check ideal conditions, as well as robustness to UNSEEN noise during training
env_config = {"motor_control_mode": "CARTESIAN_PD","task_env": "LR_COURSE_TASK",
              "observation_space_mode": "LR_COURSE_OBS"}
env_config['render'] = True
env_config['record_video'] = False
env_config['add_noise'] = False 
#env_config['competition_env'] = True

# get latest model and normalization stats, and plot 
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
model_name = get_latest_model(log_dir)
monitor_results = load_results(log_dir)
print(monitor_results)
plot_results([log_dir] , 10e10, 'timesteps', LEARNING_ALG + ' ')
plt.show() 

# reconstruct env 
env = lambda: QuadrupedGymEnv(**env_config)
env = make_vec_env(env, n_envs=1)
env = VecNormalize.load(stats_path, env)
env.training = False    # do not update stats at test time
env.norm_reward = False # reward normalization is not needed at test time

# load model
if LEARNING_ALG == "PPO":
    model = PPO.load(model_name, env)
elif LEARNING_ALG == "SAC":
    model = SAC.load(model_name, env)
#print("\nLoaded model", model_name, "\n")

obs = env.reset()
episode_reward = 0

# [TODO] initialize arrays to save data from simulation 
#
TEST_STEPS = 1000
t = np.arange(TEST_STEPS)*0.001
hopf_vars = np.zeros((TEST_STEPS,8))
joint_pos = np.zeros((TEST_STEPS,12))
base_lin_vel = np.zeros((TEST_STEPS,3))
CoT = np.zeros((TEST_STEPS,1))
legs_z = np.zeros((TEST_STEPS,4))
base_pos = np.zeros((TEST_STEPS,3))
ground_contact = np.zeros((TEST_STEPS, 4))


for i in range(TEST_STEPS):
    action, _states = model.predict(obs,deterministic=False) # sample at test time? ([TODO]: test)
    obs, rewards, dones, info = env.step(action)
    episode_reward += rewards
    CoT[i, :] = info[0]["CoT"]
    base_pos[i, :] = info[0]["base_pos"]
    
    if dones:
        
        episode_reward = 0

    # [TODO] save data from current robot states for plots 
    joint_pos[i,:] = env.envs[0].env.robot.GetMotorAngles()
    hopf_vars[i,:] = env.envs[0].env._cpg.X[0:2,:].flatten('F')
    base_lin_vel[i,:] = env.envs[0].env.robot.GetBaseLinearVelocity()
    ground_contact[i, :] = info[0]["ground_contact"]
    # To get base position, for example: env.envs[0].env.robot.GetBasePosition() 
    #
    
# [TODO] make plots:
    
env_output = QuadrupedGymEnv()
print('episode_reward', episode_reward)
print('Final base position', info[0]['base_pos'])
print("CoT: ", np.sum(CoT) / (9.81 * sum(env_output.robot.GetTotalMassFromURDF()) * base_pos[-1, 0]))
print("Average velocity: ", base_pos[-1, 0] / (TEST_STEPS * 0.001 * 10))

fig, ax = plt.subplots(1, 1)

ax.plot(base_lin_vel)
ax.set_xlabel('Steps')
ax.set_ylabel('Linear base velocity')
ax.legend(['x vel', 'y vel', 'z vel'])

fig, ax = plt.subplots(1, 1)

ax.plot(joint_pos)
ax.set_xlabel('Steps')
ax.set_ylabel('Joint position')
ax.legend(['x vel', 'y vel', 'z vel'])


fig, ax = plt.subplots(4, 1)

if TEST_STEPS > 250:
    ground_contact = ground_contact[250:TEST_STEPS]
duty_cycles = np.sum(ground_contact, axis=0) / (np.ones((1, 4)) * TEST_STEPS)
print("Duty cycles: ", duty_cycles)

plt.show()






#####################################################
# CPG states
fig = plt.figure()
plt.subplot(2, 2, 1)
plt.plot(t,hopf_vars[0,:], label='hip FR r')
plt.plot(t,hopf_vars[1,:], label='hip FR phi')
plt.grid(which='both')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t,hopf_vars[2,:], label='hip FL r')
plt.plot(t,hopf_vars[3,:], label='hip FL phi')
plt.grid(which='both')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t,hopf_vars[4,:], label='hip RR r')
plt.plot(t,hopf_vars[5,:], label='hip RR phi')
plt.grid(which='both')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t,hopf_vars[6,:], label='hip RR r')
plt.plot(t,hopf_vars[7,:], label='hip RR phi')
plt.grid(which='both')
plt.legend()
plt.show()