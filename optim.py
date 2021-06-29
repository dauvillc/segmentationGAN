"""
Defines some optimization functions, learning rate schedules, etc..
"""

import tensorflow as tf


class LinearLRSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    A linearly decreasing learning rate schedule.
    """
    def __init__(self, initial_learning_rate, total_steps, verbose=False):
        """
        -- initial_learning_rate: learning rate for the first optimization.
        -- total_steps: Number of optimisation steps (batches * batches per epoch).
        """
        self.initial_lr = initial_learning_rate
        self.steps = total_steps
        self.verbose = verbose
        self.curr_val = initial_learning_rate

    def __call__(self, step):
        self.curr_val = self.initial_lr * (1 - step / self.steps)
        return self.curr_val

    def __repr__(self):
        return self.curr_val