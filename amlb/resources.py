"""
**resources** modules exposes a singleton ``Resources`` instance providing easy access to app configuration properties,
as well as handy methods to access other resources like *automl frameworks* and *benchmark definitions*
"""
from __future__ import annotations

import copy
import logging
import os
import random
import re
import sys

from amlb.benchmarks.parser import benchmark_load
from amlb.frameworks import default_tag, load_framework_definitions
from .utils import Namespace, lazy_property, memoize, normalize_path, run_cmd, str_sanitize, touch
from .utils.config import TransformRule, config_load, transform_config
from .__version__ import __version__, _dev_version as dev


log = logging.getLogger(__name__)


class Resources:

    @staticmethod
    def _normalize(config: Namespace, replace=None):
        def nz_path(path):
            if replace is not None:
                path = path.format(**replace)
            return normalize_path(path)

        normalized = copy.copy(config)
        for k, v in config:
            if isinstance(v, Namespace):
                normalized[k] = Resources._normalize(v, replace=replace)
            elif re.search(r'_(dir|file|cmd)s?$', k):
                normalized[k] = [nz_path(p) for p in v] if isinstance(v, list) else nz_path(v)
        return normalized

    def __init__(self, config: Namespace):
        self._config = config
        common_dirs = dict(
            input=normalize_path(config.input_dir),
            output=normalize_path(config.output_dir),
            user=normalize_path(config.user_dir),
            root=normalize_path(config.root_dir),
        )
        self.config = Resources._normalize(config, replace=common_dirs)
        self.config.common_dirs = common_dirs
        log.debug("Using config:\n%s", self.config)

        # allowing to load custom modules from user directory
        sys.path.append(common_dirs['user'])
        log.debug("Extended Python sys.path to user directory: %s.", sys.path)

    @lazy_property
    def project_info(self):
        split_url = self.config.project_repository.split('#', 1)
        repo = split_url[0]
        tag = None if len(split_url) == 1 else split_url[1]
        branch = tag or 'master'
        return Namespace(
            repo=repo,
            tag=tag,
            branch=branch
        )

    @lazy_property
    def git_info(self):
        def git(cmd, defval=None):
            try:
                return run_cmd(f"git {cmd}", _log_level_=logging.DEBUG)[0].strip()
            except Exception:
                return defval

        na = "NA"
        git_version = git("--version")
        is_repo = git("rev-parse") is not None

        if git_version and is_repo:
            repo = git("remote get-url origin", na)
            branch = git("rev-parse --abbrev-ref HEAD", na)
            commit = git("rev-parse HEAD", na)
            tags = git("tag --points-at HEAD", "").splitlines()
            status = git("status -b --porcelain", "").splitlines()
        else:
            repo = branch = commit = na
            tags = status = []
        return Namespace(
            repo=repo,
            branch=branch,
            commit=commit,
            tags=tags,
            status=status
        )

    @lazy_property
    def app_version(self):
        v = __version__
        if v != dev:
            return v
        g = self.git_info
        tokens = []
        if "/openml/automlbenchmark" not in g.repo:
            tokens.append(g.repo)
        tokens.append(g.branch)
        tokens.append(g.commit[:7])
        return "{v} [{details}]".format(v=v, details=", ".join(tokens))

    def seed(self, fold=None):
        if isinstance(fold, int) and str(self.config.seed).lower() in ['auto']:
            return fold+self._seed
        else:
            return self._seed

    @lazy_property
    def _seed(self):
        if str(self.config.seed).lower() in ['none', '']:
            return None
        elif str(self.config.seed).lower() in ['auto']:
            return random.randint(1, (1 << 31) - 1)  # limiting seed to signed int32 for R frameworks
        else:
            return self.config.seed

    def framework_definition(self, name, tag=None):
        """
        :param name:
        :return: name of the framework as defined in the frameworks definition file
        """
        lname = name.lower()
        if tag is None:
            tag = default_tag
        if tag not in self._frameworks:
            raise ValueError("Incorrect tag `{}`: only those among {} are allowed.".format(tag, self.config.frameworks.tags))
        frameworks = self._frameworks[tag]
        log.debug("Available framework definitions:\n%s", frameworks)
        framework = next((f for n, f in frameworks if n.lower() == lname), None)
        # TODO: Clean up this workflow and error messaging as part of #518
        base_framework = next((f for n, f in self._frameworks[default_tag] if n.lower() == lname), None)
        if framework and framework['removed']:
            raise ValueError(f"Framework definition `{name}` has been removed from the benchmark: {framework['removed']}")
        if not framework and (base_framework and base_framework['removed']):
            raise ValueError(f"Framework definition `{name}` has been removed from the benchmark: {base_framework['removed']}")
        if not framework:
            raise ValueError(f"Incorrect framework `{name}`: not listed in {self.config.frameworks.definition_file}.")
        if framework['abstract']:
            raise ValueError(f"Framework definition `{name}` is abstract and cannot be run directly.")
        return framework, framework.name

    @lazy_property
    def _frameworks(self):
        frameworks_file = self.config.frameworks.definition_file
        return load_framework_definitions(frameworks_file, self.config)

    @memoize
    def constraint_definition(self, name):
        """
        :param name: name of the benchmark constraint definition as defined in the constraints file
        :return: a Namespace object with the constraint config (folds, cores, max_runtime_seconds, ...) for the current benchmamk run.
        """
        constraint = self._constraints[name.lower()]
        if not constraint:
            raise ValueError("Incorrect constraint definition `{}`: not listed in {}.".format(name, self.config.benchmarks.constraints_file))
        return constraint, constraint.name

    @lazy_property
    def _constraints(self):
        constraints_file = self.config.benchmarks.constraints_file
        log.info("Loading benchmark constraint definitions from %s.", constraints_file)
        if not isinstance(constraints_file, list):
            constraints_file = [constraints_file]

        constraints = Namespace()
        for ef in constraints_file:
            constraints += config_load(ef)

        for name, c in constraints:
            c.name = str_sanitize(name)

        log.debug("Available benchmark constraints:\n%s", constraints)
        constraints_lookup = Namespace()
        for name, c in constraints:
            constraints_lookup[name.lower()] = c
        return constraints_lookup

    # @memoize
    def benchmark_definition(self, name, defaults=None):
        """
        :param name: name of the benchmark as defined by resources/benchmarks/{name}.yaml, the path to a user-defined benchmark description file or a study id.
        :param defaults: defaults used as a base config for each task in the benchmark definition
        :return:
        """
        hard_defaults, tasks, benchmark_path, benchmark_name = benchmark_load(name, self.config.benchmarks.definition_dir)

        defaults = Namespace.merge(defaults, hard_defaults, Namespace(name='__defaults__'))
        for task in tasks:
            task |= defaults   # add missing keys from hard defaults + defaults
            self._validate_task(task)

        self._validate_task(defaults, lenient=True)
        defaults.enabled = False
        tasks.append(defaults)
        log.debug("Available task definitions:\n%s", tasks)
        return tasks, benchmark_name, benchmark_path

    def _validate_task(self, task, lenient=False):
        missing = []
        for conf in ['name']:
            if task[conf] is None:
                missing.append(conf)
        if not lenient and len(missing) > 0:
            raise ValueError("{missing} mandatory properties as missing in task definition {taskdef}.".format(missing=missing, taskdef=task))

        for conf in ['max_runtime_seconds', 'cores', 'folds', 'max_mem_size_mb', 'min_vol_size_mb', 'quantile_levels']:
            if task[conf] is None:
                task[conf] = self.config.benchmarks.defaults[conf]
                log.debug("Config `{config}` not set for task {name}, using default `{value}`.".format(config=conf, name=task.name, value=task[conf]))

        conf = 'id'
        if task[conf] is None:
            task[conf] = ("openml.org/t/{}".format(task.openml_task_id) if task['openml_task_id'] is not None
                          else "openml.org/d/{}".format(task.openml_dataset_id) if task['openml_dataset_id'] is not None
                          else ((task.dataset['id'] if isinstance(task.dataset, (dict, Namespace))
                                 else task.dataset if isinstance(task.dataset, str)
                                 else None) or task.name) if task['dataset'] is not None
                          else None)
            if not lenient and task[conf] is None:
                raise ValueError("task definition must contain an ID or one property "
                                 "among ['openml_task_id', 'dataset'] to create an ID, "
                                 "but task definition is {task}".format(task=str(task)))

        conf = 'metric'
        if task[conf] is None:
            task[conf] = None

        conf = 'ec2_instance_type'
        if task[conf] is None:
            i_series = self.config.aws.ec2.instance_type.series
            i_map = self.config.aws.ec2.instance_type.map
            if str(task.cores) in i_map:
                i_size = i_map[str(task.cores)]
            elif task.cores > 0:
                supported_cores = list(map(int, Namespace.dict(i_map).keys() - {'default'}))
                supported_cores.sort()
                cores = next((c for c in supported_cores if c >= task.cores), 'default')
                i_size = i_map[str(cores)]
            else:
                i_size = i_map.default
            task[conf] = '.'.join([i_series, i_size])
            log.debug("Config `{config}` not set for task {name}, using default selection `{value}`.".format(config=conf, name=task.name, value=task[conf]))

        conf = 'ec2_volume_type'
        if task[conf] is None:
            task[conf] = self.config.aws.ec2.volume_type
            log.debug("Config `{config}` not set for task {name}, using default `{value}`.".format(config=conf, name=task.name, value=task[conf]))


__INSTANCE__: Resources | None = None


def from_config(config: Namespace):
    global __INSTANCE__
    transform_config(config, _backward_compatibility_config_rules_)
    __INSTANCE__ = Resources(config)
    return __INSTANCE__


def from_configs(*configs: Namespace):
    global __INSTANCE__
    for c in configs:
        transform_config(c, _backward_compatibility_config_rules_)
    __INSTANCE__ = Resources(Namespace.merge(*configs, deep=True))
    return __INSTANCE__


def get() -> Resources:
    if __INSTANCE__ is None:
        # TODO: Instead why not do normal lazy loading pattern?
        raise RuntimeError("No configuration has been loaded yet.")
    return __INSTANCE__


def config():
    return __INSTANCE__.config


def output_dirs(root, session=None, subdirs=None, create=False):
    root = root if root is not None else '.'
    if create and not os.path.exists(root):
        touch(root, as_dir=True)

    dirs = Namespace(
        root=root,
        session=os.path.join(root, session) if session is not None else root
    )

    subdirs = ([] if subdirs is None
               else [subdirs] if isinstance(subdirs, str)
               else subdirs)

    for d in subdirs:
        dirs[d] = os.path.join(dirs.session, d)
        if create and not os.path.exists(dirs[d]):
            touch(dirs[d], as_dir=True)
    return dirs


_backward_compatibility_config_rules_ = [
    TransformRule(from_key='exit_on_error', to_key='job_scheduler.exit_on_job_failure'),
    TransformRule(from_key='parallel_jobs', to_key='job_scheduler.parallel_jobs'),
    TransformRule(from_key='max_parallel_jobs', to_key='job_scheduler.max_parallel_jobs'),
    TransformRule(from_key='delay_between_jobs', to_key='job_scheduler.delay_between_jobs'),
    TransformRule(from_key='monitoring.frequency_seconds', to_key='monitoring.interval_seconds'),
    TransformRule(from_key='aws.query_frequency_seconds', to_key='aws.query_interval_seconds'),
    TransformRule(from_key='aws.ec2.monitoring.cpu.query_frequency_seconds', to_key='aws.ec2.monitoring.cpu.query_interval_seconds'),
]
