import numpy as np
import torch

from copy import deepcopy


class ModelSaver:
    def __init__(self, model):
        self.model = model

    def save(self):
        self.state_dict = deepcopy(self.model.state_dict())

    def restore(self):
        self.model.load_state_dict(self.state_dict)


class EarlyStopping:
    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError

    def next_step(self):
        raise NotImplementedError

    def should_save(self):
        raise NotImplementedError

    def should_stop(self):
        raise NotImplementedError


class NoEarlyStopping(EarlyStopping):
    def __init__(self):
        super().__init__()

    def reset(self):
        pass

    def next_step(self):
        pass

    def should_stop(self):
        return False

    def should_save(self):
        return False


class NoImprovementStopping(EarlyStopping):
    def __init__(self, validation_fn, mode='min', patience=10, tolerance=0.0, relative=False):
        super().__init__()
        self.validation_fn = validation_fn
        self.mode = mode
        self.patience = patience
        self.tolerance = tolerance
        self.relative = relative
        self.reset()

        if mode not in ['min', 'max']:
            raise ValueError(f"Mode should be either 'min' or 'max' (got {mode} instead).")

        if relative:
            if mode == 'min':
                self._is_better = lambda new, best: new < best - (best * tolerance)
            if mode == 'max':
                self._is_better = lambda new, best: new > best + (best * tolerance)
        else:
            if mode == 'min':
                self._is_better = lambda new, best: new < best - tolerance
            if mode == 'max':
                self._is_better = lambda new, best: new > best + tolerance

    def reset(self):
        self._best_value = self.validation_fn()
        self._num_bad_epochs = 0
        self._time_to_save = False

    def next_step(self):
        last_value = self.validation_fn()
        if self._is_better(last_value, self._best_value):
            self._time_to_save = True
            self._best_value = last_value
            self._num_bad_epochs = 0
        else:
            self._num_bad_epochs += 1

    def should_save(self):
        if self._time_to_save:
            self._time_to_save = False
            return True
        else:
            return False

    def should_stop(self):
        return self._num_bad_epochs > self.patience
