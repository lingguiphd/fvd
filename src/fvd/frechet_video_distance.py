import numpy as np
from scipy import linalg
from tqdm import tqdm
from pathlib import Path

import torch
from torch.nn.functional import interpolate

from .pytorch_i3d import InceptionI3d


def preprocess(videos, target_resolution):
    reshaped_videos = videos.permute(0, 4, 1, 2, 3)
    size = [reshaped_videos.size()[2]] + list(target_resolution)
    resized_videos = interpolate(reshaped_videos, size=size, mode='trilinear', align_corners=False)
    scaled_videos = 2 * resized_videos / 255. - 1
    return scaled_videos


def get_statistics(activations):
    mean = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mean, cov


def calculate_fvd_from_activations(first_activations, second_activations, eps=1e-10):
    f_mean, f_cov = get_statistics(first_activations)
    s_mean, s_cov = get_statistics(second_activations)

    diff = f_mean - s_mean

    sqrt_cov = linalg.sqrtm(f_cov.dot(s_cov))
    if not np.isfinite(sqrt_cov).all():
        print("Sqrtm calculation produces singular values;",
              "adding %s to diagonal of cov estimates." % eps)
        offset = np.eye(f_cov.shape[0]) * eps
        sqrt_cov = linalg.sqrtm((f_cov + offset).dot(s_cov + offset))
    sqrt_cov = sqrt_cov.real

    return diff.dot(diff) + np.trace(f_cov + s_cov - 2 * sqrt_cov)


def batch_generator(data, batch_size):
    n = data.size()[0]
    indices = np.random.permutation(n)

    for i in tqdm(range(0, n, batch_size)):
        batch_indices = indices[i:i+batch_size]
        yield data[batch_indices]


def get_activations(data, model, batch_size=10):
    activations = []
    for batch in batch_generator(data, batch_size):
        activations.append(model(batch).squeeze().detach().numpy())
    return np.vstack(activations)


def frechet_video_distance(first_set_of_videos: torch.Tensor, second_set_of_videos: torch.Tensor, path_to_model_weights: Path) -> float:
    i3d = InceptionI3d(400, in_channels=3)
    i3d.load_state_dict(torch.load(path_to_model_weights))
    i3d.train(False)

    print("Calculating activations for the first set of videos...")
    first_activations = get_activations(preprocess(first_set_of_videos, (224, 224)), i3d)
    print("Calculating activations for the second set of videos...")
    second_activations = get_activations(preprocess(second_set_of_videos, (224, 224)), i3d)
    return calculate_fvd_from_activations(first_activations, second_activations)


class FrechetVideoDistance:
    """ FVD 指标计算器。"""
    
    def __init__(self, path_to_model_weights: Path) -> None:
        """ 
        初始化。
        
        Args:
            path_to_model_weights (Path): I3D 模型权重文件的路径。
        """
        self.i3d = InceptionI3d(400, in_channels=3)
        self.i3d.load_state_dict(torch.load(path_to_model_weights))
        self.i3d.train(False)
        
    def compute(self, first_set_of_videos: torch.Tensor, second_set_of_videos: torch.Tensor) -> float:
        """
        计算两个视频集合之间的 FVD。
        
        Args:
            first_set_of_videos (torch.Tensor): 第一个视频集合，形状为 (N, T, H, W, C)。
            second_set_of_videos (torch.Tensor): 第二个视频集合，形状为 (M, T, H, W, C)。
            
        Returns:
            float: 两个视频集合之间的 FVD。
        """
        # 获取两个视频集合的激活值
        first_activations = get_activations(preprocess(first_set_of_videos, (224, 224)), self.i3d)
        second_activations = get_activations(preprocess(second_set_of_videos, (224, 224)), self.i3d)
        # 计算 FVD
        fvd = calculate_fvd_from_activations(first_activations, second_activations)
        return fvd
