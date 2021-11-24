    
import time
#import wandb
import os
import numpy as np
from itertools import chain
import torch
import imageio

from onpolicy.utils.util import update_linear_schedule
from base_runner import Runner

level_num=2

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for level_index in range(level_num):
                    self.trainer[level_index].policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                import pdb
                pdb.set_trace()
                # Obser reward and next obs

                actions_env_in = np.concatenate(actions_env,axis=1)
                obs, rewards, dones, infos = self.envs.step(actions_env_in)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        form_rews = []
                        avoi_rews = []
                        nav_rews = []
                        for info in infos:
                            if 'individual_reward' in info[agent_id].keys():
                                idv_rews.append(info[agent_id]['individual_reward'])
                            if 'formation_reward' in info[agent_id].keys():
                                form_rews.append(info[agent_id]['formation_reward'])
                            if 'avoidance_reward' in info[agent_id].keys():
                                avoi_rews.append(info[agent_id]['avoidance_reward'])
                            if 'navigation_reward' in info[agent_id].keys():
                                nav_rews.append(info[agent_id]['navigation_reward'])
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})

                        train_infos[agent_id].update({'formation_reward': np.mean(form_rews)})
                        train_infos[agent_id].update({'avoidance_reward': np.mean(avoi_rews)})
                        train_infos[agent_id].update({'navigation_reward': np.mean(nav_rews)})
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        obs_list = obs.tolist() # [rollout * [2]]
        obs_n_level_1 = []
        obs_n_level_2 = []
        for single_env_obs_array in obs_list:
            obs_n_level_1.append(single_env_obs_array[0])
            obs_n_level_2.append(single_env_obs_array[1])


        self.buffer[0].obs[0] = np.array(obs_n_level_1).copy()
        self.buffer[1].obs[0] = np.array(obs_n_level_2).copy()


    @torch.no_grad()
    def collect(self, step):
        values_for_each = []
        actions_for_each = []
        temp_actions_env_for_each = []
        action_log_probs_for_each = []
        rnn_states_for_each = []
        rnn_states_critic_for_each = []

        for level_index in range(level_num):
            self.trainer[level_index].prep_rollout()
            value, action, action_log_prob, rnn_states, rnn_states_critic \
                = self.trainer[level_index].policy.get_actions(np.concatenate(self.buffer[level_index].share_obs[step]),
                            np.concatenate(self.buffer[level_index].obs[step]),
                            np.concatenate(self.buffer[level_index].rnn_states[step]),
                            np.concatenate(self.buffer[level_index].rnn_states_critic[step]),
                            np.concatenate(self.buffer[level_index].masks[step]))

            # [self.envs, agents, dim]
            values = np.array(np.split(_t2n(value), self.n_rollout_threads))
            actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
            action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
            rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
            rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
            
            # [agents, envs, dim]
            # values.append(_t2n(value))
            # action = _t2n(action)
            # rearrange action
            if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[0].shape):
                    uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                    if i == 0:
                        actions_env = uc_actions_env
                    else:
                        actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
            elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
                actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
            elif self.envs.action_space[0].__class__.__name__ == 'Box':
                actions_env = actions
            else:
                raise NotImplementedError

            values_for_each.append(values) 
            actions_for_each.append(actions)
            action_log_probs_for_each.append(action_log_probs)
            rnn_states_for_each.append(rnn_states)
            rnn_states_critic_for_each.append(rnn_states_critic)

            # action_env = np.array([[np.argmax(item) for item in actions_env[j]] for j in range(actions_env.shape[0])])

            temp_actions_env_for_each.append(actions_env)



        return values_for_each, actions_for_each, action_log_probs_for_each, rnn_states_for_each, rnn_states_critic_for_each, temp_actions_env_for_each # 所有的东西都是(len=2)的列表

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        
        obs_level_1 = obs[:,0:3,:]
        obs_level_2 = obs[:,3:,:]

        obs_level_1 = obs[:,0:3,:]


        share_obs_level_1 = []
        share_obs_level_2 = []


        share_obs_level_1 = []
        for o in obs_level_1:
            share_obs_level_1.append(list(chain(*o)))
        share_obs_level_1 = np.array(share_obs_level_1)

        share_obs_level_2 = []
        for o in obs_level_2:
            share_obs_level_2.append(list(chain(*o)))
        share_obs_level_2 = np.array(share_obs_level_2)

        self.buffer[0].insert(share_obs_level_1, obs_level_1, rnn_states, rnn_states_critic, actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(level_num):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(level_num):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        envs = self.envs
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                temp_actions_env = []
                for agent_id in range(level_num):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    elif self.envs.action_space[0].__class__.__name__ == 'Box':
                        action_env = action
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            for agent_id in range(level_num):
                average_episode_rewards = np.mean(np.sum(np.array(episode_rewards)[:, :, agent_id, :], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + 'render.gif', all_frames, duration=self.all_args.ifi)