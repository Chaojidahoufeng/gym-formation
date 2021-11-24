import numpy as np
from scipy.spatial.distance import directed_hausdorff

from formation_gym.scenario import BaseScenario
from formation_gym.core import World, Agent, Landmark

'''
partly centralized formation for large scale
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications
'''

class Scenario(BaseScenario):
    def make_world(self, all_args, num_agents = 3, num_landmarks = 3, episode_length = 100):
        # world properties
        self.args = all_args
        num_agents_level_1 = 3
        num_agents_level_2 = 3 # 3 level-two agents for each level-one agent
        num_lamdmarks_level_1 = 3
        num_landmarks_level_2 = 3
        world = World()
        world.world_length = episode_length
        world.dim_c = 2 # communication channel
        world.collaborative = True
        # agent properties

        world.num_agents_level_1 = num_agents_level_1
        world.num_agents_level_2 = num_agents_level_2

        world.agents_level_1 = [Agent() for i in range(num_agents_level_1)]
        world.agents_level_2 = [[Agent() for j in range(num_agents_level_2)] for i in range(num_agents_level_1)]
        world.agents = world.agents_level_1

        for i, agent in enumerate(world.agents_level_1):
            agent.name = 'level1 '+'agent %d' % i
            agent.collide = False
            agent.silent = True
            agent.size = 0.08
        # landmark properties
        for i, agent_list in enumerate(world.agents_level_2):
            world.agents = world.agents + agent_list
            for j, agent in enumerate(agent_list):
                agent.name = 'level2 ' + 'agent %d_%d' % (i, j)
                agent.collide = False
                agent.silent = True
                agent.size = 0.06
            
        world.landmarks_level_1 = [Landmark() for i in range(num_lamdmarks_level_1)]
        world.landmarks_level_2 = [[Landmark() for j in range(num_landmarks_level_2)] for i in range(num_lamdmarks_level_1)]
        world.landmarks = world.landmarks_level_1

        for i, landmark in enumerate(world.landmarks_level_1):
            landmark.name = 'level1 '+'landmarks %d' % i
            landmark.collide = False 
            landmark.movable = False
            landmark.size = 0.04

        for i, landmark_list in enumerate(world.landmarks_level_2):
            world.landmarks = world.landmarks + landmark_list
            for j, landmark in enumerate(landmark_list):
                landmark.name = 'level2 '+'landmarks %d_%d' % (i, j)
                landmark.collide = False
                landmark.movable = False
                landmark.size = 0.02

        # initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent, world):
        # agent pos & communication
        if 'level1' in agent.name:
            entity_pos = []
            for entity in world.landmarks_level_1:
                entity_pos.append(entity.state.p_pos)
            
            other_pos = []
            comm = []
            for other in world.agents_level_1:
                if other is agent: continue
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)

            center = np.mean(other_pos, 0)
        elif 'level2' in agent.name:
            agent_cluster_num = int(agent.name.split(' ')[-1].split('_')[0])
            entity_pos = []
            for entity in world.landmarks_level_2[agent_cluster_num]:
                entity_pos.append(entity.state.p_pos)

            other_pos = []
            comm = []

            agent_cluster_num = int(agent.name.split(' ')[-1].split('_')[0])
            agent_list = world.agents_level_2[agent_cluster_num]
            for other in agent_list:
                if other is agent: continue
                comm.append(other.state.c)
                other_pos.append(other.state.p_pos - agent.state.p_pos)
            
            center = world.agents_level_1[agent_cluster_num].state.p_pos

        else:
            raise NotImplementedError()

        # print(np.concatenate([agent.state.p_vel]+entity_pos + other_pos + comm + center))

        return np.concatenate([agent.state.p_vel]+entity_pos + other_pos + comm + center)

    def reward(self, agent, world):
        rew = 0
        if 'level1' in agent.name:
            u = [a.state.p_pos for a in world.agents_level_1]
            v = [l.state.p_pos for l in world.landmarks_level_1]
            delta = np.mean(u, 0) - np.mean(v, 0)
            u = u - np.mean(u, 0)
            v = v - np.mean(v, 0)
            rew = -max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    
        elif 'level2' in agent.name:
            agent_cluster_num = int(agent.name.split(' ')[-1].split('_')[0])
            u = [a.state.p_pos for a in world.agents_level_2[agent_cluster_num]]
            v = [l.state.p_pos for l in world.landmarks_level_2[agent_cluster_num]]
            delta = np.mean(u, 0) - np.mean(v, 0)
            u = u - np.mean(u, 0)
            v = v - np.mean(v, 0)
            formation_rew = -max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
            rew = 0 + formation_rew * self.args.form_factor

            delta_cluster_cent_vs_leader = np.mean(u, 0) - world.agents_level_1[agent_cluster_num].state.p_pos
            dis_rew = - np.linalg.norm(delta_cluster_cent_vs_leader, ord=2)

            rew += dis_rew * self.args.dis_factor

            # print('formation_rew: ' + str(formation_rew))
            # print('dis_rew:' + str(dis_rew))
        # # change landmark pos and color
        # for i in range(len(world.landmarks)):
        #     delta = [0, 0]
        #     world.landmarks[i].state.p_pos += delta
            # dist = min([np.linalg.norm(a.state.p_pos - world.landmarks[i].state.p_pos) for a in world.agents])
            # if dist <= 0.2: world.landmarks[i].color = np.array([0, 0.6, 0])
        # self.set_bound(world)
        if agent.collide:
            for a in world.agents:
                if agent!=a and self.is_collision(a, agent):
                    rew -= 1
        return rew

    def reset_world(self, world):
        # agent
        for agent in world.agents_level_1:
            agent.color = np.array([0.35, 0.35, 0.85])
            agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)

        for i, agent_list in enumerate(world.agents_level_2):
            for j, agent in enumerate(agent_list):
                agent.color = np.array([1.0-i*0.2, i*0.2, i*0.2])
                agent.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
                agent.state.p_vel = np.zeros(world.dim_p)
                agent.state.c = np.zeros(world.dim_c)
        # landmark
        for landmark in world.landmarks_level_1:
            landmark.color = np.array([0.25, 0.25, 0.25])
            landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)
            landmark.state.p_vel = np.zeros(world.dim_p)

        for i, landmark_list in enumerate(world.landmarks_level_2):
            for j, landmark in enumerate(landmark_list):
                landmark.color = np.array([0.25, 0.25, 0.25])
                landmark.state.p_pos = np.random.uniform(-1, +1, world.dim_p)/2
                landmark.state.p_vel = np.zeros(world.dim_p)

        for i, landmark_list in enumerate(world.landmarks_level_2):
            v = [l.state.p_pos for l in landmark_list]
            delta = world.landmarks_level_1[i].state.p_pos - np.mean(v, 0)
            for j, landmark in enumerate(landmark_list):
                landmark.state.p_pos += delta

    def benchmark_data(self, agent, world):
        # get data to debug
        rew = self.reward(agent, world)
        collisions = 0
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    collisions += 1
        min_dists = 0
        occupied_landmarks = 0
        for l in world.landmarks:
            dists = [np.linalg.norm(a.state.p_pos - l.state.p_pos) for a in world.agents]
            min_dists += min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        return {
            'reward': rew, 
            'collisions': collisions, 
            'min_dists': min_dists, 
            'occupied_landmarks': occupied_landmarks
        }

    def is_collision(self, agent1, agent2):
        dist = np.linalg.norm(agent1.state.p_pos - agent2.state.p_pos)
        return dist < (agent1.size + agent2.size)/2

    def set_bound(self, world):
        for agent in world.agents:
            agent.state.p_pos = np.clip(agent.state.p_pos, [-2, -2], [2, 2])

