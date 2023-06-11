"""
Schedulers for curriculum batch sampler
"""
import warnings
import mmcv

from mmcv import Registry
from typing import List, Iterable, Union


CURRICULUM_SCHEDULERS = Registry("curriculum_schedulers")


class CurriculumSchedulerBase:
    def __init__(self, seq_names: List[str]):
        self._sel_seqs = None

        assert len(seq_names) > 0, 'seq_names can not be empty.'
        self._seq_names = seq_names

    @property
    def selected_sequences(self):
        return self._sel_seqs

    def select_with_ids(self, ids: Iterable[int], seq_names: List[str] = None):
        """
        Select sequences with indexes
        :param seq_names: from which to select
        :param ids: the indexes
        :return:
        """
        seq_names = self._seq_names if seq_names is None else seq_names
        seqs = [seq_names[idx] for idx in filter(lambda x: x < len(seq_names), ids)]
        return seqs

    def set_sel_seqs(self, sel_seqs: List[str]):
        self._sel_seqs = sel_seqs
        return self._sel_seqs

    def __call__(self, curr_iters: int, seq_names: List[str] = None) -> List[str]:
        """
        Return selected sequences
        :param curr_iters:
        :return:
        """
        raise NotImplementedError

    def __repr__(self):
        props = filter(lambda x: not x.startswith('_'), self.__dict__.keys())
        desc = ','.join([f'{k}: {self.__dict__[k]}' for k in props])
        return f'{type(self).__name__}, ({desc}).'


@CURRICULUM_SCHEDULERS.register_module(name='uniform')
class UniformCurriculumScheduler(CurriculumSchedulerBase):
    def __init__(self, n_init_seqs: int, n_step_seqs: int, n_step_iters: int, seq_names: List[str],
                 n_max_seqs: int = 1000):
        super(UniformCurriculumScheduler, self).__init__(seq_names)

        self.n_init_seqs = n_init_seqs
        self.n_step_seqs = n_step_seqs
        self.n_step_iters = n_step_iters
        self.n_max_seqs = n_max_seqs

        self.set_sel_seqs(self.select_with_ids(range(self.n_init_seqs)))

    def __call__(self, curr_iters: int, seq_names: List[str] = None):
        n_lim_seqs = (curr_iters // self.n_step_iters) * self.n_step_seqs + self.n_init_seqs
        n_lim_seqs = min(n_lim_seqs, self.n_max_seqs)
        return self.set_sel_seqs(self.select_with_ids(range(n_lim_seqs), seq_names))


@CURRICULUM_SCHEDULERS.register_module(name='power')
class PowerCurriculumScheduler(CurriculumSchedulerBase):
    def __init__(self, n_init_seqs: int, power: float, seq_names: List[str], n_max_seqs: int = 1000):
        super(PowerCurriculumScheduler, self).__init__(seq_names)

        self.n_init_seqs = n_init_seqs
        self.power = power
        self.n_max_seqs = n_max_seqs

        self.set_sel_seqs(self.select_with_ids(range(self.n_init_seqs)))

        if self.power > 1.0:
            warnings.warn(f'The power ({self.power}) is too big.')

    def __call__(self, curr_iters: int, seq_names: List[str] = None):
        n_lim_seqs = int(curr_iters ** self.power) + self.n_init_seqs
        n_lim_seqs = min(n_lim_seqs, self.n_max_seqs)
        return self.set_sel_seqs(self.select_with_ids(range(n_lim_seqs), seq_names))


@CURRICULUM_SCHEDULERS.register_module(name='range')
class RangeCurriculumScheduler(CurriculumSchedulerBase):
    def __init__(self, n_lim_seqs: int, seq_names: List[str]):
        super(RangeCurriculumScheduler, self).__init__(seq_names)

        self.n_lim_seqs = n_lim_seqs

        self.set_sel_seqs(self.select_with_ids(range(self.n_lim_seqs)))

    def __call__(self, curr_iters: int, seq_names: List[str] = None):
        return self.set_sel_seqs(self.select_with_ids(range(self.n_lim_seqs), seq_names))


@CURRICULUM_SCHEDULERS.register_module(name='joint')
class JointCurriculumScheduler(CurriculumSchedulerBase):
    def __init__(self, scheduler: CurriculumSchedulerBase, seq_names: List[str]):
        super(JointCurriculumScheduler, self).__init__(seq_names)

        assert isinstance(scheduler, CurriculumSchedulerBase)
        self.scheduler = scheduler

    @property
    def selected_sequences(self):
        return self.scheduler.selected_sequences

    def __call__(self, curr_iters: int, seq_names: List[str] = None):
        return self.selected_sequences


@CURRICULUM_SCHEDULERS.register_module(name='fixed')
class FixedCurriculumScheduler(CurriculumSchedulerBase):
    def __init__(self, seq_list: Union[str, List[str]], seq_names: List[str]):
        super(FixedCurriculumScheduler, self).__init__(seq_names)

        if isinstance(seq_list, str):
            seq_list = list(set(mmcv.list_from_file(seq_list)))

        self.set_sel_seqs(seq_list)
        self.num_sequences = len(self._sel_seqs)

    def __call__(self, curr_iters: int, seq_names: List[str] = None):
        return self.selected_sequences


@CURRICULUM_SCHEDULERS.register_module(name='composed')
class ComposedCurriculumScheduler(CurriculumSchedulerBase):
    def __init__(self, sub_schedulers: List[dict], seq_names: List[str]):
        super(ComposedCurriculumScheduler, self).__init__(seq_names)

        assert len(sub_schedulers) > 0, 'sub_schedulers can not be empty.'
        schedulers = []
        for config in sub_schedulers:
            assert 'seq_names' not in config, f'seq_names can not be specified.'
            config['seq_names'] = seq_names
            schedulers.append(CURRICULUM_SCHEDULERS.build(config))
        self.schedulers = schedulers
        self.set_sel_seqs(self.schedulers[0].selected_sequences)

    def __call__(self, curr_iters: int, seq_names: List[str] = None):
        sel_seqs = seq_names
        for scheduler in self.schedulers:
            sel_seqs = scheduler(curr_iters, sel_seqs)
        return self.set_sel_seqs(sel_seqs)
