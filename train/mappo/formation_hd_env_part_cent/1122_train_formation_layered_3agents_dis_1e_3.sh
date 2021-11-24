#!/bin/sh
env="MPE"
#scenario="formation_hd_env_layered"  # simple_speaker_listener # simple_reference
scenario="formation_hd_env_layered"
num_agents_level_1=3
num_agents_level_2=3
algo="rmappo"
exp="1122_train_formation_layered_3agents_dis_1e_3"
seed_max=1

echo "env is ${env}, scenario is ${scenario}, algo is ${algo}, exp is ${exp}, max seed is ${seed_max}"
for seed in `seq ${seed_max}`;
do
    echo "seed is ${seed}:"
    python train_formation_layered.py \
    --dis_factor 0.001 \
    --use_valuenorm \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --scenario_name ${scenario} \
    --num_agents_level_1 ${num_agents_level_1} \
    --num_agents_level_2 ${num_agents_level_2} \
    --seed ${seed} \
    --n_training_threads 1 \
    --n_rollout_threads 128 \
    --num_mini_batch 1 \
    --episode_length 25 \
    --num_env_steps 20000000 \
    --ppo_epoch 10 \
    --use_ReLU \
    --gain 0.01 \
    --lr 7e-4 \
    --critic_lr 7e-4 \
    --wandb_name "chaojidahoufeng" \
    --user_name "jc-bao"
done