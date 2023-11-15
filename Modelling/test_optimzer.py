import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.optim import Adam
import matplotlib.pyplot as plt


STEPS = 100


class Scheduler(_LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    @staticmethod
    def __calc_lr(step, dim_embed, warmup_steps):
        return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    def get_lr(self) -> float:
        lr = self.__calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups


optimizer = Adam(params=[torch.tensor(1)], lr=1, betas=(0.9, 0.99), amsgrad=False)

# Use a scheduler of your choice below.
# Great for debugging your own schedulers!
scheduler_0 = Scheduler(optimizer=optimizer, dim_embed=512, warmup_steps=4000)
scheduler_1 = CosineAnnealingLR(optimizer=optimizer, T_max=STEPS)
scheduler_2 = CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=STEPS, T_mult=1)

lr_0 = []
lr_1 = []
lr_2 = []

for i in range(400):
    optimizer.step()
    if i < 100 - 1:
        lr_0.append(scheduler_0.get_lr())
        lr_1.append(scheduler_1.get_lr())
        lr_2.append(scheduler_2.get_lr())
    else:
        lr_0.append(scheduler_0.get_last_lr())
        lr_1.append(scheduler_1.get_last_lr())
        lr_2.append(scheduler_2.get_last_lr())
    scheduler_0.step()
    scheduler_1.step()
    scheduler_2.step()

fig, ax = plt.subplots(nrows=1, ncols=3)
ax[0].plot(lr_0)
ax[1].plot(lr_1)
ax[2].plot(lr_2)
plt.show()