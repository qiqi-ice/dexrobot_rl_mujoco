"""Subprocess-based vectorized environment for parallel MuJoCo simulation."""

import multiprocessing as mp
from enum import Enum

import numpy as np


class _Cmd(Enum):
    RESET = "reset"
    STEP = "step"
    CLOSE = "close"


def _worker(env_fn, pipe, parent_pipe):
    parent_pipe.close()
    env = env_fn()
    try:
        while True:
            cmd, data = pipe.recv()
            if cmd == _Cmd.RESET:
                obs, info = env.reset()
                pipe.send((obs, info))
            elif cmd == _Cmd.STEP:
                obs, reward, terminated, truncated, info = env.step(data)
                done = terminated or truncated
                if done:
                    obs, _ = env.reset()
                pipe.send((obs, float(reward), bool(done), info))
            elif cmd == _Cmd.CLOSE:
                env.close()
                break
    finally:
        pipe.close()


class SubprocVecEnv:
    """Run N envs in parallel subprocesses.

    Each env is created by calling env_fns[i](). Observations are stacked into
    arrays of shape (num_envs, ...).
    """

    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        ctx = mp.get_context("fork")
        self._parent_pipes = []
        self._processes = []
        for fn in env_fns:
            parent_pipe, child_pipe = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(fn, child_pipe, parent_pipe), daemon=True)
            p.start()
            child_pipe.close()
            self._parent_pipes.append(parent_pipe)
            self._processes.append(p)

        # Infer action_dim from first env
        dummy = env_fns[0]()
        self.action_dim = dummy.action_dim
        obs, _ = dummy.reset()
        self.obs_dim = obs.shape[0]
        dummy.close()

    def reset(self):
        for pipe in self._parent_pipes:
            pipe.send((_Cmd.RESET, None))
        results = [pipe.recv() for pipe in self._parent_pipes]
        obs = np.stack([r[0] for r in results], axis=0)
        infos = [r[1] for r in results]
        return obs, infos

    def step(self, actions):
        """actions: np.ndarray of shape (num_envs, action_dim)"""
        for pipe, action in zip(self._parent_pipes, actions):
            pipe.send((_Cmd.STEP, action))
        results = [pipe.recv() for pipe in self._parent_pipes]
        obs = np.stack([r[0] for r in results], axis=0)
        rewards = np.array([r[1] for r in results], dtype=np.float32)
        dones = np.array([r[2] for r in results], dtype=np.float32)
        infos = [r[3] for r in results]
        return obs, rewards, dones, infos

    def close(self):
        for pipe in self._parent_pipes:
            try:
                pipe.send((_Cmd.CLOSE, None))
            except BrokenPipeError:
                pass
        for p in self._processes:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
