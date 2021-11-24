#!/bin/sh
env="MPE"
#scenario="formation_hd_env_layered"  # simple_speaker_listener # simple_reference
scenario="formation_hd_env_layered"
num_agents_level_1=3
num_agents_level_2=3
algo="rmappo"
exp="1123_render_formation_layered_share_3agents_dis_5e_2"
seed_max=1

echo "env is ${env}"
for seed in `seq ${seed_max}`
do
    # CUDA_VISIBLE_DEVICES=1 python render_formation.py --save_gifs --share_policy --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --scenario_name ${scenario} --num_agents ${num_agents} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 --use_render --episode_length 25 --render_episodes 5 --model_dir "./results/MPE/formation_hd_env/rmappo/check/run15/models"
    python render_formation_layered_share.py \
    --dis_factor 0.001 \
    --use_valuenorm \
    --save_gifs \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents_level_1 ${num_agents_level_1} \
    --num_agents_level_2 ${num_agents_level_2} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 1 \
    --use_render \
    --use_ReLU \
    --episode_length 50 \
    --render_episodes 1 \
    --model_dir "/home/yanyz/data/gym-formation/train/mappo/results/MPE/formation_hd_env_layered/rmappo/1123_train_formation_layered_share_3agents_dis_5e_2/run1/models" \
    --gif_dir './results/3agents/gif'
done