import torch
import os
import typing


_EXT_NAME = '.pt'


class KeyPointPool:
    def __init__(self, dir_name: str):
        self._key_points = {}

        # scan files and append key points
        for f in os.scandir(dir_name):
            if f.is_file() and f.name.endswith(_EXT_NAME):
                self._key_points[f.name[: -len(_EXT_NAME)]] = torch.load(f.path)
        print(f'{len(self._key_points)} key point items are loaded.')

    def __call__(self, seq_name: str, keys: typing.List[str]):
        key_points = self._key_points[seq_name]
        points = torch.stack(
            [key_points[k] for k in keys],
            dim=0
        )  # (n, 3)
        return points
