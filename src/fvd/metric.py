# 深度学习指标
from torchmetrics import Metric
from torchmetrics.utilities.plot import _AX_TYPE as AX_TYPE, _PLOT_OUT_TYPE as PLOT_OUT_TYPE
# 类型提示
from typing import Optional, Sequence
# 文件路径
from pathlib import Path
# 张量
import torch
# 模型
from .pytorch_i3d import InceptionI3d
# 工具函数
from .frechet_video_distance import preprocess, get_activations, calculate_fvd_from_activations


class FrechetVideoDistance(Metric):
    """ FVD 指标计算器。"""
    
    higher_is_better: Optional[bool] = False
    is_differentiable: Optional[bool] = False
    full_state_update: Optional[bool] = False
    plot_lower_bound: Optional[float] = 0.0
    
    i3d: torch.nn.Module
    feature_network: str = "i3d"
    
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

    def set_dtype(self, dst_type: str | torch.dtype) -> Metric:
        """Transfer all metric state to specific dtype. Special version of standard `type` method.

        Arguments:
            dst_type: the desired type as ``torch.dtype`` or string

        """
        out = super().set_dtype(dst_type)
        if isinstance(out.inception, InceptionI3d):
            out.inception._dtype = dst_type
        return out
    
    def plot(
        self, 
        val: Optional[torch.Tensor, Sequence[torch.Tensor]] = None, 
        ax: Optional[AX_TYPE] = None,
    ) -> PLOT_OUT_TYPE:
        """Plot a single or multiple values from the metric.

        Args:
            val: Either a single result from calling `metric.forward` or `metric.compute` or a list of these results.
                If no value is provided, will automatically call `metric.compute` and plot that result.
            ax: An matplotlib axis object. If provided will add plot to that axis

        Returns:
            Figure and Axes object

        Raises:
            ModuleNotFoundError:
                If `matplotlib` is not installed

        .. plot::
            :scale: 75

            >>> # Example plotting a single value
            >>> import torch
            >>> from torchmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> metric.update(imgs_dist1, real=True)
            >>> metric.update(imgs_dist2, real=False)
            >>> fig_, ax_ = metric.plot()

        .. plot::
            :scale: 75

            >>> # Example plotting multiple values
            >>> import torch
            >>> from torchmetrics.image.fid import FrechetInceptionDistance
            >>> imgs_dist1 = lambda: torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            >>> imgs_dist2 = lambda: torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            >>> metric = FrechetInceptionDistance(feature=64)
            >>> values = [ ]
            >>> for _ in range(3):
            ...     metric.update(imgs_dist1(), real=True)
            ...     metric.update(imgs_dist2(), real=False)
            ...     values.append(metric.compute())
            ...     metric.reset()
            >>> fig_, ax_ = metric.plot(values)

        """
        return self._plot(val, ax)
