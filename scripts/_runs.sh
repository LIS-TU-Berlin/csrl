
export CUDA_NUM_GPUS=`nvidia-smi -L | wc -l`

python scripts/CSRL.py -m +hydra_runs=true first_gpu=0 seed=1,2,3,4,5 \
scenario=double_sphere,allegro_down


#table1
#scenario=double_sphere,panda_sphere \
#Gym.reward_dense_radius=0.0,1.0 CSRL.start_goal_mode=CC,RC,RR,CCi,RCi,RRi,CCic,RCic,RRic

#table2
#scenario=double_sphere,panda_sphere \
#CSRL.RL_method=sbTD3 CSRL.start_goal_mode=CC,CCi,CCic

#table2
#scenario=double_sphere,panda_sphere \
#TD7.HER_ratio=0.0 CSRL.start_goal_mode=CC,CCi,CCic

#table2
#scenario=double_sphere,panda_sphere \
#CSRL.RL_method=sbPPO CSRL.start_goal_mode=CC,CCi,CCic

#table3
#scenario=sphere_cube,panda_cube,allegro_cube,allegroDown_cube \
#CSRL.start_goal_mode=CC,CCi,CCic



#Gym.reward_dense_radius=0.0 CSRL.start_goal_mode=CC,RC,RR,CCi,RCi,RRi

#CSRL.start_goal_mode=CC,RC,RR,CCi,RCi,RRi,CCic,RCic,RRic

#CSRL.start_goal_mode=rvC,rvR

#CSRL.T_curr=0 CSRL.start_goal_mode=CC,RC,RR,CCi,RCi,RRi 
       
#TD7.HER_ratio=0.0 CSRL.start_goal_mode=CCic,RCic,RRic

#Gym.reward_dense_radius=0.0 CSRL.start_goal_mode=CCi,RRi,RCi

#Gym.reward_dense_radius=0.0 TD7.HER_ratio=0.0 CSRL.T_curr=0 CSRL.start_goal_mode=CC,RC,RR,CCi,RCi,RRi
       
       
#T_curr=0,100_000 \
#TD7.HER_ratio=0.0,1.0

#TD7.critic_hdim=512 \
#TD7.actor_hdim=512 \
#TD7.zs_dim=1024 \
#TD7.enc_hdim=1024 \
#Gym.warp_worlds=2000,5000 \
#TD7.train_steps=20,50 \
#Gym.warp_worlds=5000 \
#TD7.learning_rate=1e-3,3e-4 \
#TD7.batch_size=5000,10000 \
#TD7.buffer_size=1000000,2000000,5000000 \
#seed=1,2

#Gym.warp_worlds=1000,2000,5000
#TD7.train_steps=10,20,50
#TD7.learning_rate=1e-3,3e-4
#TD7.buffer_size=2000000,4000000,6000000
#TD7.batch_size=2000,5000,10000
#seed=1,2

#seed=0,1
#TD7.HER_ratio=0.0,1.0
#TD7.train_steps=5,10 #fewer is better, it seems
#Gym.warp_worlds=2000,5000
#hydra.launcher.n_jobs=8

