import numpy as np

def traj_to_dataset(trajectories, del_rgb = False):
    dataset = dict()
    dataset['obs'] = trajectories['obs']
    dataset['actions'] = trajectories['actions']
    re = trajectories['rewards']
    dataset['rewards_unnorm'] = re
    dataset['rewards'] = (re - np.mean(re, axis=0)) / (np.std(re, axis=0) + 1e-8)
    dataset['next_obs'] = trajectories['next_obs']
    if isinstance(dataset['next_obs'], dict):
        # obs_mode='pointcloud'
        dataset['next_pcd'] = dataset['next_obs'].pop('pointcloud')
        next_s = trajectories['next_obs']['state']
        dataset['ds'] = (next_s - np.mean(next_s, axis=0)) / (np.std(next_s, axis=0) + 1e-8)
        if del_rgb:
            del dataset['obs']['pointcloud']['rgb']
            del dataset['next_pcd']['rgb']
    else: # obs_mode='state'
        raise NotImplementedError

    return dataset

def get_statistics(buffer_list):
    datas = []
    for i in range(len(buffer_list)):
        if len(buffer_list[i]): datas.append(buffer_list[i].get_all())

    ds = np.concatenate([data['next_obs']['state'] - data['obs']['state'] for data in datas], axis=0)
    re = np.concatenate([data['rewards_unnorm'] for data in datas], axis=0)
    ds_mean = np.mean(ds, axis=0)
    ds_std = np.std(ds, axis=0)
    re_mean = np.mean(re, axis=0)
    re_std = np.std(re, axis=0)

    return ds_mean, ds_std, re_mean, re_std