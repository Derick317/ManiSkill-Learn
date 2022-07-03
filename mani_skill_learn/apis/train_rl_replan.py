import itertools
import os
import os.path as osp
import time
import re
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import csv, numpy as np

from mani_skill_learn.env import ReplayMemory
from mani_skill_learn.env import save_eval_statistics
from mani_skill_learn.utils.data import dict_to_str, get_shape, is_seq_of, to_np
from mani_skill_learn.utils.meta import get_logger, get_total_memory, td_format, Config
from mani_skill_learn.utils.torch import TensorboardLogger, save_checkpoint
from mani_skill_learn.utils.math import split_num
from mani_skill_learn.utils.meta import traj_to_dataset, get_statistics
from mani_skill_learn.methods.mbrl import REPLANV2, REPLAN

class EpisodicStatistics:
    def __init__(self, num_procs):
        self.num_procs = num_procs
        self.current_lens = np.zeros(num_procs)
        self.current_rewards = np.zeros(num_procs)
        self.history_rewards = np.zeros(num_procs)
        self.history_lens = np.zeros(num_procs)
        self.history_counts = np.zeros(num_procs)

    def push(self, rewards, dones):
        n, running_steps = split_num(len(dones), self.num_procs)
        j = 0
        for i in range(n):
            for _ in range(running_steps[i]):
                self.current_lens[i] += 1
                self.current_rewards[i] += rewards[j]
                if dones[j]:
                    self.history_rewards[i] += self.current_rewards[i]
                    self.history_lens[i] += self.current_lens[i]
                    self.history_counts[i] += 1
                    self.current_rewards[i] = 0
                    self.current_lens[i] = 0
                j += 1

    def reset_history(self):
        self.history_lens *= 0
        self.history_rewards *= 0
        self.history_counts *= 0

    def reset_current(self):
        self.current_rewards *= 0
        self.current_lens *= 0

    def get_mean(self):
        num_episode = np.clip(np.sum(self.history_counts), a_min=1E-5, a_max=1E10)
        return np.sum(self.history_lens) / num_episode, np.sum(self.history_rewards) / num_episode

    def print_current(self):
        print(self.current_lens, self.current_rewards)

    def print_history(self):
        print(self.history_lens, self.history_rewards, self.history_counts)

class EveryNSteps:
    def __init__(self, interval=None):
        self.interval = interval
        self.next_value = interval

    def reset(self):
        self.next_value = self.interval

    def check(self, x):
        if self.interval is None:
            return False
        sign = False
        while x >= self.next_value:
            self.next_value += self.interval
            sign = True
        return sign

    def standard(self, x):
        return int(x // self.interval) * self.interval

def train_rl(agent, rollout, evaluator, env_cfg, buffers: dict, on_policy, work_dir, 
             total_steps=1000000, warm_steps=10000, n_traj_onPol=1, n_traj_rand=1, 
             n_updates=1, n_checkpoint=None, n_eval=None, init_replay_buffers=None,
             init_replay_with_split=None, eval_cfg=None, reset_hp=False,
             replicate_init_buffer=1, num_trajs_per_demo_file=-1):
    logger = get_logger(env_cfg.env_name)

    import torch
    from mani_skill_learn.utils.torch import get_cuda_info
    for buffer in buffers.values():
        buffer.reset()

    if init_replay_buffers is not None and init_replay_buffers != '':
        raise NotImplementedError
        replay.restore(init_replay_buffers, replicate_init_buffer, num_trajs_per_demo_file)
        logger.info(f'Initialize buffer with {len(replay)} samples')
    if init_replay_with_split is not None:
        raise NotImplementedError
        assert is_seq_of(init_replay_with_split) and len(init_replay_with_split) == 2
        # For mani skill only
        from mani_skill.utils.misc import get_model_ids_from_yaml
        folder_root = init_replay_with_split[0]
        model_split_file = get_model_ids_from_yaml(init_replay_with_split[1])
        if init_replay_with_split[1] is None:
            files = [str(_) for _ in Path(folder_root).glob('*.h5')]
        else:
            files = [str(_) for _ in Path(folder_root).glob('*.h5') if re.split('[_-]', _.name)[1] in model_split_file]
        replay.restore(files, replicate_init_buffer, num_trajs_per_demo_file)

    tf_logs = ReplayMemory(total_steps)
    tf_logs.reset()
    tf_logger = TensorboardLogger(work_dir)

    checkpoint_dir = osp.join(work_dir, 'models')
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(agent)
    if rollout is not None:
        logger.info(f'Rollout state dim: {get_shape(rollout.recent_obs)}, action dim: {len(rollout.random_action())}')
        rollout.reset()
        episode_statistics = EpisodicStatistics(1)
        total_episodes = 0
    else:
        # Batch RL
        for buffer in buffers.values():
            if 'obs' not in buffer.memory:
                logger.error('Empty replay buffer for Batch RL!')
                exit(0)
            logger.info(f'State dim: {get_shape(buffer["obs"])}, action dim: {buffer["actions"].shape[-1]}')

    check_eval = EveryNSteps(n_eval)
    check_checkpoint = EveryNSteps(n_checkpoint)
    check_tf_log = EveryNSteps(1000)

    steps = 0

    if warm_steps > 0:
        logger.info(f"Start warming!")
        assert not on_policy
        assert rollout is not None
        # valOnPol and valRand
        trajectories = rollout.forward_with_policy(None, warm_steps // 2)[0]
        episode_statistics.push(trajectories['rewards'], trajectories['episode_dones'])
        dataset = traj_to_dataset(trajectories, del_rgb=agent.del_rgb)
        buffers["valOnPol"].push_batch(**dataset)
        buffers["valRand"].push_batch(**dataset)
        rollout.reset()
        # trainRand
        trajectories = rollout.forward_with_policy(None, warm_steps)[0]
        episode_statistics.push(trajectories['rewards'], trajectories['episode_dones'])
        dataset = traj_to_dataset(trajectories, del_rgb=agent.del_rgb)
        buffers["trainRand"].push_batch(**dataset)
        buffers["trainOnPol"].push_batch(**dataset)
        rollout.reset()
        episode_statistics.reset_current()
        steps = warm_steps * 3 // 2
        check_eval.check(steps)
        check_checkpoint.check(steps)
        check_tf_log.check(steps)
        logger.info(f"Finish {steps} warm-up steps!")

    total_updates = 0
    begin_time = datetime.now()
    max_ETA_len = None
    header = ('iter', 'epoch', 'train_ds', 'train_chamfer',	'train_rewards', 'train_total', 
              'valOnPol_ds', 'valOnPol_chamfer', 'valOnPol_rewards', 'valOnPol_total', 
              'valRand_ds', 'valRand_chamfer', 'valRand_rewards', 'valRand_total')
    with open(work_dir + "/loss_statistics.csv", 'w', encoding='utf-8', newline='') as f:  
        write = csv.writer(f)
        write.writerow(header)

    for iteration_id in itertools.count(1):
        if reset_hp:
            try:
                args = Config.fromfile("configs/update_hyperparameter.py")
                if 'num_action_sequences' in args and agent.policy.N != args.num_action_sequences:
                    agent.policy.N = args.num_action_sequences
                    logger.info(f"\"Num_action_sequences\" of the policy changes to {agent.policy.N}.")
                if 'horizon' in args and agent.policy.horizon != args.horizon:
                    agent.policy.horizon = args.horizon
                    logger.info(f"\"Horizon\" of the policy changes to {agent.policy.horizon}.")
                if 'mppi_beta' in args and agent.policy.mppi_beta != args.mppi_beta:
                    agent.policy.mppi_beta = args.mppi_beta
                    logger.info(f"\"Beta\" of the policy changes to {agent.policy.mppi_beta}.")
                if 'n_traj_onPol' in args and n_traj_onPol != args.n_traj_onPol:
                    n_traj_onPol = args.n_traj_onPol
                    logger.info(f"\"n_traj_onPol\" changes to {n_traj_onPol}.")
                if 'n_traj_rand' in args and n_traj_rand != args.n_traj_rand:
                    n_traj_rand = args.n_traj_rand
                    logger.info(f"\"n_traj_rand\" changes to {n_traj_rand}.")
                if 'nEpoch' in args and agent.nEpoch != args.nEpoch:
                    if isinstance(agent, REPLANV2):
                        agent.change_nEpoch(args.nEpoch)
                    else: agent.nEpoch = args.nEpoch
                    logger.info(f"\"nEpoch\" of the agent changes to {agent.nEpoch}.")
            except Exception as e:
                print(e)

        tf_logs.reset()
        if rollout:
            episode_statistics.reset_history()

        if on_policy:
            for buffer in buffers.values():
                buffer.reset()

        train_dict = {}
        model_dict = {}
        print_dict = OrderedDict()

        update_time = 0
        time_begin_episode = time.time()

        """
        Train the model
        """
        tmp_time = time.time()

        # Start to train
        # print("Start training the model ...", end='')
        train_loss, valOnPol_loss, valRand_loss = agent.train_model(buffers)
        for key in train_loss[-1][1]:
            model_dict['train_' + key] = float(to_np(train_loss[-1][1][key]))
        for key in valOnPol_loss[-1][1]:
            model_dict['valOnPol_' + key] = float(to_np(valOnPol_loss[-1][1][key]))
        for key in valRand_loss[-1][1]:
            model_dict['valRand_' + key] = float(to_np(valRand_loss[-1][1][key]))

        start_epoch, end_epoch = train_loss[0][0], train_loss[-1][0]
        writelist = []
        for j in range(end_epoch - start_epoch + 1):
            writelist.append([iteration_id, j + start_epoch, train_loss[j][1]['ds'], train_loss[j][1]['next_pcd'],
                train_loss[j][1]['rewards'], train_loss[j][1]['total'], valOnPol_loss[j][1]['ds'],
                valOnPol_loss[j][1]['next_pcd'], valOnPol_loss[j][1]['rewards'], valOnPol_loss[j][1]['total'],
                valRand_loss[j][1]['ds'], valRand_loss[j][1]['next_pcd'], valRand_loss[j][1]['rewards'], 
                valRand_loss[j][1]['total']])
        with open(work_dir + "/loss_statistics.csv", 'a', encoding='utf-8', newline='') as f:
            write = csv.writer(f)
            for j in range(end_epoch - start_epoch + 1):
                write.writerow(writelist[j])
        update_time = time.time() - tmp_time

        """
        Perform rollouts
        """
        print("Start performing rollouts ...", end='\r')
        if n_traj_onPol > 0:
            # For online RL
            cnt_episodes = 0
            tmp_time = time.time()
            # Perform rollout with MPC policy
            trajectories, infos = rollout.forward_n_episodes(agent.policy, n_traj_onPol)
            for i in range(n_traj_onPol):
                traj = trajectories[i]
                episode_statistics.push(traj['rewards'], traj['episode_dones'])
                cnt_episodes += np.sum(traj['episode_dones'].astype(np.int32))

                dataset = traj_to_dataset(traj, del_rgb=agent.del_rgb)
                if (i % 10) == 0: 
                    buffers['valOnPol'].push_batch(**dataset)
                else:
                    buffers['trainOnPol'].push_batch(**dataset)
                steps += traj['actions'].shape[0]

            # Perform rollout with random policy
            trajectories, infos = rollout.forward_n_episodes(None, n_traj_rand)
            for i in range(n_traj_rand):
                traj = trajectories[i]
                episode_statistics.push(traj['rewards'], traj['episode_dones'])
                cnt_episodes += np.sum(traj['episode_dones'].astype(np.int32))

                dataset = traj_to_dataset(traj, del_rgb=agent.del_rgb)
                buffers['trainRand'].push_batch(**dataset)
                steps += traj['actions'].shape[0]
            collect_sample_time = time.time() - tmp_time

            assert cnt_episodes == n_traj_onPol + n_traj_rand
            total_episodes += cnt_episodes
            train_dict['total_episode'] = int(total_episodes)
            train_dict['episode_time'] = time.time() - time_begin_episode
            train_dict['collect_sample_time'] = collect_sample_time

            print_dict['episode_length'], print_dict['episode_reward'] = episode_statistics.get_mean()
        else:
            # For offline RL
            raise NotImplementedError
            tf_logs.reset()
            for i in range(n_updates):
                steps += 1
                total_updates += 1
                tmp_time = time.time()
                tf_logs.push(**agent.update_parameters(replay, updates=total_updates))
                update_time += time.time() - tmp_time

        """
        Save information
        """
        train_dict['update_time'] = update_time
        train_dict['total_updates'] = int(total_updates)
        train_dict['valOnPol_buffer_size'] = len(buffers['valOnPol'])
        train_dict['trainOnPol_buffer_size'] = len(buffers['trainOnPol'])
        train_dict['valRand_buffer_size'] = len(buffers['valRand'])
        train_dict['trainRand_buffer_size'] = len(buffers['trainRand'])
        train_dict['memory'] = get_total_memory('G', True)
        train_dict['cuda_mem'] = get_total_memory('G', True)

        train_dict.update(get_cuda_info(device=torch.cuda.current_device()))

        # print_dict.update(tf_logs.tail_mean(n_updates))
        print_dict['memory'] = get_total_memory('G', False)
        print_dict['train_buffer_size'] = len(buffers['trainOnPol']) + len(buffers['trainRand'])
        print_dict['val_buffer_size'] = len(buffers['valOnPol']) + len(buffers['valRand'])
        print_dict.update(get_cuda_info(device=torch.cuda.current_device(), number_only=False))
        print_dict.update(model_dict)
        print_info = dict_to_str(print_dict)

        percentage = f'{(steps / total_steps) * 100:.0f}%'
        passed_time = td_format(datetime.now() - begin_time)
        ETA = td_format((datetime.now() - begin_time) * (total_steps / (steps - warm_steps) - 1))
        if max_ETA_len is None:
            max_ETA_len = len(ETA)

        logger.info(f'{steps}/{total_steps}({percentage}) Passed time:{passed_time} ETA:{ETA} {print_info}')
        if check_tf_log.check(steps):
            train_dict.update(dict(print_dict))
            tf_logger.log(train_dict, n_iter=steps, eval=False)

        if check_checkpoint.check(steps):
            standardized_ckpt_step = check_checkpoint.standard(steps)
            model_path = osp.join(checkpoint_dir, f'model_{standardized_ckpt_step}.ckpt')
            logger.info(f'Save model at step: {steps}.The model will be saved at {model_path}')
            if isinstance(agent, REPLAN):
                agent.to_normal()
                save_checkpoint(agent, model_path)
                agent.recover_data_parallel()
            elif isinstance(agent, REPLANV2):
                agent.save_checkpoint(model_path)
            else:
                raise NotImplementedError
        if check_eval.check(steps):
            standardized_eval_step = check_eval.standard(steps)
            logger.info(f'Begin to evaluate at step: {steps}. '
                        f'The evaluation info will be saved at eval_{standardized_eval_step}')
            eval_dir = osp.join(work_dir, f'eval_{standardized_eval_step}')
            agent.eval()
            torch.cuda.empty_cache()
            lens, rewards, finishes, selected_id, target_indexes = evaluator.run(agent, **eval_cfg, work_dir=eval_dir)
            torch.cuda.empty_cache()
            save_eval_statistics(eval_dir, lens, rewards, finishes, selected_id, target_indexes, logger)
            agent.train()

            eval_dict = {}
            eval_dict['mean_length'] = np.mean(lens)
            eval_dict['std_length'] = np.std(lens)
            eval_dict['mean_reward'] = np.mean(rewards)
            eval_dict['std_reward'] = np.std(rewards)
            tf_logger.log(eval_dict, n_iter=steps, eval=True)

        if steps >= total_steps:
            break
    if n_checkpoint:
        print(f'Save checkpoint at final step {total_steps}')
        if isinstance(agent, REPLAN):
            agent.to_normal()
            save_checkpoint(agent, osp.join(checkpoint_dir, f'model_{total_steps}.ckpt'))
        elif isinstance(agent, REPLANV2):
            agent.save_checkpoint(osp.join(checkpoint_dir, f'model_{total_steps}.ckpt'))
        else:
            raise NotImplementedError
