# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Union

import torch
from torch.cuda.amp.grad_scaler import OptState
from torch.optim import Optimizer

from pytorch_lightning.plugins.precision_plugin import PrecisionPlugin


class LightningGradScaler(torch.cuda.amp.GradScaler):

    def __init__(self, trainer, init_scale=2. ** 16, growth_factor=2.0, backoff_factor=0.5, growth_interval=2000,
                 enabled=True):

        self.trainer = trainer

        super().__init__(init_scale, growth_factor, backoff_factor, growth_interval, enabled)

    def step(self, optimizer, *args, **kwargs):
        """
        :meth:`step` carries out the following two operations:

        1.  Internally invokes ``unscale_(optimizer)`` (unless :meth:`unscale_` was explicitly called for ``optimizer``
            earlier in the iteration).  As part of the :meth:`unscale_`, gradients are checked for infs/NaNs.
        2.  If no inf/NaN gradients are found, invokes ``optimizer.step()`` using the unscaled
            gradients.  Otherwise, ``optimizer.step()`` is skipped to avoid corrupting the params.

        ``*args`` and ``**kwargs`` are forwarded to ``optimizer.step()``.

        Returns the return value of ``optimizer.step(*args, **kwargs)``.

        Arguments:
            optimizer (torch.optim.Optimizer):  Optimizer that applies the gradients.
            args:  Any arguments.
            kwargs:  Any keyword arguments.

        .. warning::
            Closure use is not currently supported.
        """
        if (not self._enabled):
            return optimizer.step(*args, **kwargs)

        if "closure" in kwargs:
            raise RuntimeError("Closure use is not currently supported if GradScaler is enabled.")
        with self.trainer.profiler.profile("amp scaler check scale growth"):
            self._check_scale_growth_tracker("step")

        optimizer_state = self._per_optimizer_states[id(optimizer)]

        if optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("step() has already been called since the last update().")

        retval = None

        if (hasattr(optimizer, "_step_supports_amp_scaling") and optimizer._step_supports_amp_scaling):
            # This optimizer has customized scale-handling logic, so we can call optimizer.step() directly.
            # The contract with custom optimizers is that their step() should accept an additional,
            # optional grad_scaler kwarg.  We append self to the kwargs so the custom optimizer has full information:
            # it can query its own state, invoke unscale_ on itself, etc
            retval = optimizer.step(*args, **dict(kwargs, grad_scaler=self))
            optimizer_state["stage"] = OptState.STEPPED
            return retval

        with self.trainer.profiler.profile("amp scaler unscale"):
            if optimizer_state["stage"] is OptState.READY:
                self.unscale_(optimizer)

        assert len(optimizer_state["found_inf_per_device"]) > 0, "No inf checks were recorded for this optimizer."
        with self.trainer.profiler.profile("amp scaler optim get inf sum"):
            items = (v.item() for v in optimizer_state["found_inf_per_device"].values())
        with self.trainer.profiler.profile("amp scaler set devices"):
            print(set(v.device for v in optimizer_state["found_inf_per_device"].values()))
        with self.trainer.profiler.profile("amp scaler vals"):
            print(set(v.item() for v in optimizer_state["found_inf_per_device"].values()))
        with self.trainer.profiler.profile("amp scaler optim step inf sum"):
            num_infs = sum(items)
        print("NUM INF DEVICE", len(optimizer_state["found_inf_per_device"].values()))
        with self.trainer.profiler.profile("amp scaler optim step inf loop"):
            if not num_infs:
                with self.trainer.profiler.profile("amp scaler optim step"):
                    retval = optimizer.step(*args, **kwargs)

        optimizer_state["stage"] = OptState.STEPPED

        return retval


class NativeAMPPlugin(PrecisionPlugin):

    def __init__(self, trainer=None):
        """
        Integrates native amp into Lightning's internals.
        """
        self.trainer = trainer

    def connect(self, model, optimizers):
        return model, optimizers

    def backward(self, closure_loss, optimizer, opt_idx, *args, **kwargs):
        with self.trainer.profiler.profile("scale loss"):
            closure_loss = self.trainer.scaler.scale(closure_loss)

        automatic_optimization = self.trainer.train_loop.automatic_optimization

        # do backward pass
        if automatic_optimization:
            model = self.trainer.get_model()
            model.backward(closure_loss, optimizer, opt_idx)
        else:
            closure_loss.backward(*args, **kwargs)

        # once backward has been applied, release graph
        closure_loss = closure_loss.detach()
        with self.trainer.profiler.profile("unscale loss"):
            # unscale gradient to allow analyze within `on_after_backward`
            if not self.trainer.train_loop.should_accumulate() and automatic_optimization:
                self.trainer.scaler.unscale_(optimizer)

        return closure_loss

    def training_step(self, fx, args):
        with torch.cuda.amp.autocast():
            output = fx(*args)
        return output

    @property
    def scaler(self):
        return LightningGradScaler(self.trainer)

    def clip_gradients(self, grad_clip_val: Union[int, float], optimizer: Optimizer, norm_type: float):
        model = self.trainer.get_model()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_val, norm_type=norm_type)
