"""
Reference: https://github.com/allenai/allennlp/blob/main/allennlp/training/learning_rate_schedulers/cosine.py
Author: AllenNLP
"""

import torch
import numpy as np

class CosineWithWarmRestarts(torch.optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warm restarts.

    Parameters
    ----------
    optionimizer : torch.optionim.Optimizer

    T_max : int
        The maximum number of iterations within the first cycle.

    eta_min : float, optionional (default: 0)
        The minimum learning rate.

    last_epoch : int, optionional (default: -1)
        The index of the last epoch.

    """
    def __init__(self,
                optionimizer: torch.optim.Optimizer,
                T_max: int,
                eta_min: float = 0.,
                last_epoch: int = -1,
                factor: float = 1.) -> None:
        self.T_max = T_max
        self.eta_min = eta_min
        self.factor = factor
        self._last_restart: int = 0
        self._cycle_counter: int = 0
        self._cycle_factor: float = 1.
        self._updated_cycle_len: int = T_max
        self._initialized: bool = False
        super().__init__(optionimizer, last_epoch)

    def get_lr(self):
        if not self._initialized:
            self._initialized = True
            return self.base_lrs

        step = self.last_epoch + 1
        self._cycle_counter = step - self._last_restart

        lr_scheduler = [
            (
             self.eta_min + 1/2 * (lr - self.eta_min) *
                (
                 1 + np.cos(np.pi * (self._cycle_counter % self._updated_cycle_len) / self._updated_cycle_len)
                )
            ) for lr in self.base_lrs
        ]

        if self._cycle_counter % self._updated_cycle_len == 0:
            self._cycle_factor *= self.factor
            self._cycle_counter = 0
            self._updated_cycle_len = int(self._cycle_factor * self.T_max)
            self._last_restart = step
        return lr_scheduler