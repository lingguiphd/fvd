import numpy as np
from scipy import linalg
from tqdm import tqdm
from pathlib import Path
from typing import Generator

import torch
from torch.nn.functional import interpolate

from .pytorch_i3d import InceptionI3d


def preprocess(videos: torch.Tensor, target_resolution: tuple[int, int]) -> torch.Tensor:
    reshaped_videos = videos.permute(0, 4, 1, 2, 3)
    size = [reshaped_videos.size()[2]] + list(target_resolution)
    resized_videos: torch.Tensor = interpolate(reshaped_videos, size=size, mode='trilinear', align_corners=False)
    scaled_videos: torch.Tensor = 2 * resized_videos / 255. - 1
    return scaled_videos


def get_statistics(activations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ 
    获取激活值的均值和协方差矩阵。
    
    Args:
        activations (np.ndarray): 激活值，形状为 (N, D)。
        
    Returns:
        tuple[np.ndarray, np.ndarray]: 均值和协方差矩阵。
    """
    mean = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mean, cov


def calculate_fvd_from_activations(first_activations: np.ndarray, second_activations: np.ndarray, eps: float = 1e-10) -> float:
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


def batch_generator(data: torch.Tensor, batch_size: int) -> Generator[torch.Tensor, None, None]:
    n = data.size()[0]
    indices = np.random.permutation(n)

    for i in tqdm(range(0, n, batch_size)):
        batch_indices = indices[i:i+batch_size]
        yield data[batch_indices]


def get_activations(data: torch.Tensor, model: InceptionI3d, batch_size: int = 10) -> np.ndarray:
    activations: list[np.ndarray] = []
    for batch in batch_generator(data, batch_size):
        output: torch.Tensor = model(batch)
        activations.append(output.squeeze().cpu().detach().numpy())
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