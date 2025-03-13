"""
**resources** modules exposes a singleton ``Resources`` instance providing easy access to app configuration properties,
as well as handy methods to access other resources like *automl frameworks* and *benchmark definitions*
"""

from __future__ import annotations

import copy
import dataclasses
import logging
import os
import random
import re
import sys
from functools import cache, cached_property

from amlb.benchmarks.parser import benchmark_load
from amlb.frameworks import default_tag, load_framework_definitions
from .frameworks.definitions import TaskConstraint
from .utils import (
    Namespace,
    normalize_path,
    run_cmd,
    str_sanitize,
    touch,
)
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
            elif re.search(r"_(dir|file|cmd)s?$", k):
                normalized[k] = (
                    [nz_path(p) for p in v] if isinstance(v, list) else nz_path(v)
                )
        return normalized

    def __init__(self, config: Namespace):
        """
        Initializes the Resources instance using the specified configuration.

        Normalizes directory paths for input, output, user, and root directories and
        updates the configuration with these common directories. Additionally, appends
        the normalized user directory to Python's module search path to enable custom
        module loading.

        Parameters:
            config: A Namespace containing configuration settings, including attributes
                'input_dir', 'output_dir', 'user_dir', and 'root_dir'.
        """
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
        sys.path.append(common_dirs["user"])
        log.debug("Extended Python sys.path to user directory: %s.", sys.path)

    @cached_property
    def project_info(self):
        """
        Extracts repository information from the project's repository URL.

        Splits the URL on the '#' character to separate the repository path from an optional tag.
        If a tag is present, it is used as the branch name; otherwise, the branch defaults to "master".
        Returns a Namespace with 'repo', 'tag', and 'branch' attributes.
        """
        split_url = self.config.project_repository.split("#", 1)
        repo = split_url[0]
        tag = None if len(split_url) == 1 else split_url[1]
        branch = tag or "master"
        return Namespace(repo=repo, tag=tag, branch=branch)

    @cached_property
    def git_info(self):
        """
        Retrieves Git repository details.

        This method executes various Git commands to obtain repository URL, branch name,
        commit hash, tags at HEAD, and the status output. If Git is not available or the
        current directory is not a repository, it returns default placeholder values.

        Returns:
            Namespace: An object with the following attributes:
                repo (str): Remote repository URL or "NA".
                branch (str): Current branch name or "NA".
                commit (str): Latest commit hash or "NA".
                tags (list[str]): Tags pointing at HEAD, or an empty list.
                status (list[str]): Git status output as a list of lines, or an empty list.
        """

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
            repo=repo, branch=branch, commit=commit, tags=tags, status=status
        )

    @cached_property
    def app_version(self):
        """
        Returns the formatted application version string.

        If the version indicates a development build, appends Git metadata—such as a non-default
        repository URL, branch, and abbreviated commit hash—to the base version. Otherwise, returns the
        version unchanged.
        """
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
        """
        Compute and return a seed for random number generation.

        If a fold index is provided and the configuration seed is set to "auto",
        returns the sum of the fold value and the base seed; otherwise, returns the
        base seed from the configuration.

        Args:
            fold (int, optional): Fold index used to offset the base seed for variability
                when the seed is auto-generated.

        Returns:
            int: The computed seed value.
        """
        if isinstance(fold, int) and str(self.config.seed).lower() in ["auto"]:
            return fold + self._seed
        else:
            return self._seed

    @cached_property
    def _seed(self):
        """
        Compute the seed value for random number generation.

        Returns None if the configured seed is "none" or empty. If the seed is set to "auto", generates a random integer between 1 and 2^31 - 1 (ensuring compatibility with signed 32-bit R frameworks). Otherwise, returns the explicitly configured seed.
        """
        if str(self.config.seed).lower() in ["none", ""]:
            return None
        elif str(self.config.seed).lower() in ["auto"]:
            return random.randint(
                1, (1 << 31) - 1
            )  # limiting seed to signed int32 for R frameworks
        else:
            return self.config.seed

    def framework_definition(self, name, tag=None):
        """
        Retrieve a valid framework definition by name and tag.

        This function searches for and returns a framework definition based on the provided
        framework name and an optional tag (defaulting to a preset tag if omitted). The search
        is case-insensitive and validates that the framework is active (i.e., not removed) and
        not abstract. If the provided tag is invalid, or if the framework is removed, abstract,
        or not found, a ValueError is raised.

        Args:
            name: The name of the framework to retrieve, case-insensitively.
            tag: Optional; the tag under which to search for the framework. Defaults to a preset tag.

        Returns:
            A tuple containing the framework definition and its identifier.

        Raises:
            ValueError: If the tag is invalid, if the framework has been removed, is abstract,
                        or if the framework is not listed in the definition file.
        """
        lname = name.lower()
        if tag is None:
            tag = default_tag
        if tag not in self._frameworks:
            raise ValueError(
                "Incorrect tag `{}`: only those among {} are allowed.".format(
                    tag, self.config.frameworks.tags
                )
            )
        frameworks = self._frameworks[tag]
        log.debug("Available framework definitions:\n%s", frameworks)
        framework = next((f for n, f in frameworks if n.lower() == lname), None)
        base_framework = next(
            (f for n, f in self._frameworks[default_tag] if n.lower() == lname), None
        )
        if framework and framework["removed"]:
            raise ValueError(
                f"Framework definition `{name}` has been removed from the benchmark: {framework['removed']}"
            )
        if not framework and (base_framework and base_framework["removed"]):
            raise ValueError(
                f"Framework definition `{name}` has been removed from the benchmark: {base_framework['removed']}"
            )
        if not framework:
            raise ValueError(
                f"Incorrect framework `{name}`: not listed in {self.config.frameworks.definition_file}."
            )
        if framework["abstract"]:
            raise ValueError(
                f"Framework definition `{name}` is abstract and cannot be run directly."
            )
        return framework, framework.name

    @cached_property
    def _frameworks(self):
        """
        Load framework definitions from the configuration file.

        Extracts the file path from self.config.frameworks.definition_file and loads the framework
        definitions using load_framework_definitions, returning the parsed definitions.
        """
        frameworks_file = self.config.frameworks.definition_file
        return load_framework_definitions(frameworks_file, self.config)

    @cache
    def constraint_definition(self, name: str) -> TaskConstraint:
        """
        Retrieve the benchmark constraint definition for a given name.

        This method performs a case-insensitive lookup in the loaded constraints and converts the
        resulting configuration into a TaskConstraint instance. It raises a ValueError if no matching
        constraint is found.

        Args:
            name (str): The name of the benchmark constraint definition as specified in the constraints file.

        Returns:
            TaskConstraint: An instance containing configuration details such as folds, cores, and max_runtime_seconds.

        Raises:
            ValueError: If the requested constraint definition is not found.
        """
        constraint = self._constraints[name.lower()]
        if not constraint:
            raise ValueError(
                "Incorrect constraint definition `{}`: not listed in {}.".format(
                    name, self.config.benchmarks.constraints_file
                )
            )
        return TaskConstraint(**Namespace.dict(constraint))

    @cached_property
    def _constraints(self):
        """
        Loads benchmark constraint definitions from configuration files.

        Processes the constraint files specified in the configuration by merging their
        contents, sanitizing each constraint name, and constructing a lookup Namespace
        with keys converted to lowercase.
        """
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

    def benchmark_definition(self, name: str, defaults: TaskConstraint | None = None):
        return self._benchmark_definition(name, self.config, defaults)

    def _benchmark_definition(
        self, name: str, config_: Namespace, defaults: TaskConstraint | None = None
    ):
        """
        :param name: name of the benchmark as defined by resources/benchmarks/{name}.yaml, the path to a user-defined benchmark description file or a study id.
        :param defaults: defaults used as a base config for each task in the benchmark definition
        :return:
        """
        file_defaults, tasks, benchmark_path, benchmark_name = benchmark_load(
            name, config_.benchmarks.definition_dir
        )
        if defaults is not None:
            defaults = Namespace(**dataclasses.asdict(defaults))
        defaults_ = Namespace.merge(
            defaults, file_defaults, Namespace(name="__defaults__")
        )
        for task in tasks:
            task |= defaults_  # add missing keys from hard defaults + defaults
            Resources._validate_task(task, config_)

        Resources._validate_task(defaults, config_, lenient=True)
        defaults_.enabled = False
        tasks.append(defaults_)
        log.debug("Available task definitions:\n%s", tasks)
        return tasks, benchmark_name, benchmark_path

    @staticmethod
    def _validate_task(task: Namespace, config_: Namespace, lenient: bool = False):
        missing = []
        for conf in ["name"]:
            if task[conf] is None:
                missing.append(conf)
        if not lenient and len(missing) > 0:
            raise ValueError(
                "{missing} mandatory properties as missing in task definition {taskdef}.".format(
                    missing=missing, taskdef=task
                )
            )

        for conf in [
            "max_runtime_seconds",
            "cores",
            "folds",
            "max_mem_size_mb",
            "min_vol_size_mb",
            "quantile_levels",
        ]:
            if task[conf] is None:
                task[conf] = config_.benchmarks.defaults[conf]
                log.debug(
                    "Config `{config}` not set for task {name}, using default `{value}`.".format(
                        config=conf, name=task.name, value=task[conf]
                    )
                )

        if task["metric"] is None:
            task["metric"] = None

        conf = "id"
        if task[conf] is None:
            task[conf] = (
                "openml.org/t/{}".format(task.openml_task_id)
                if task["openml_task_id"] is not None
                else "openml.org/d/{}".format(task.openml_dataset_id)
                if task["openml_dataset_id"] is not None
                else (
                    (
                        task.dataset["id"]
                        if isinstance(task.dataset, (dict, Namespace))
                        else task.dataset
                        if isinstance(task.dataset, str)
                        else None
                    )
                    or task.name
                )
                if task["dataset"] is not None
                else None
            )
            if not lenient and task[conf] is None:
                raise ValueError(
                    "task definition must contain an ID or one property "
                    "among ['openml_task_id', 'dataset'] to create an ID, "
                    "but task definition is {task}".format(task=str(task))
                )

        conf = "ec2_instance_type"
        if task[conf] is None:
            i_series = config_.aws.ec2.instance_type.series
            i_map = config_.aws.ec2.instance_type.map
            if str(task.cores) in i_map:
                i_size = i_map[str(task.cores)]
            elif task.cores > 0:
                supported_cores = list(
                    map(int, Namespace.dict(i_map).keys() - {"default"})
                )
                supported_cores.sort()
                cores = next((c for c in supported_cores if c >= task.cores), "default")
                i_size = i_map[str(cores)]
            else:
                i_size = i_map.default
            task[conf] = ".".join([i_series, i_size])
            log.debug(
                "Config `{config}` not set for task {name}, using default selection `{value}`.".format(
                    config=conf, name=task.name, value=task[conf]
                )
            )

        conf = "ec2_volume_type"
        if task[conf] is None:
            task[conf] = config_.aws.ec2.volume_type
            log.debug(
                "Config `{config}` not set for task {name}, using default `{value}`.".format(
                    config=conf, name=task.name, value=task[conf]
                )
            )


__INSTANCE__: Resources | None = None


def from_configs(*configs: Namespace):
    global __INSTANCE__
    for c in configs:
        transform_config(c, _backward_compatibility_config_rules_)
    __INSTANCE__ = Resources(Namespace.merge(*configs, deep=True))
    return __INSTANCE__


def get() -> Resources:
    if __INSTANCE__ is None:
        raise RuntimeError("No configuration has been loaded yet.")
    return __INSTANCE__


def config():
    if __INSTANCE__ is None:
        raise RuntimeError("No configuration has been loaded yet.")
    return __INSTANCE__.config


def output_dirs(root, session=None, subdirs=None, create=False):
    root = root if root is not None else "."
    if create and not os.path.exists(root):
        touch(root, as_dir=True)

    dirs = Namespace(
        root=root, session=os.path.join(root, session) if session is not None else root
    )

    subdirs = (
        [] if subdirs is None else [subdirs] if isinstance(subdirs, str) else subdirs
    )

    for d in subdirs:
        dirs[d] = os.path.join(dirs.session, d)
        if create and not os.path.exists(dirs[d]):
            touch(dirs[d], as_dir=True)
    return dirs


_backward_compatibility_config_rules_ = [
    TransformRule(from_key="exit_on_error", to_key="job_scheduler.exit_on_job_failure"),
    TransformRule(from_key="parallel_jobs", to_key="job_scheduler.parallel_jobs"),
    TransformRule(
        from_key="max_parallel_jobs", to_key="job_scheduler.max_parallel_jobs"
    ),
    TransformRule(
        from_key="delay_between_jobs", to_key="job_scheduler.delay_between_jobs"
    ),
    TransformRule(
        from_key="monitoring.frequency_seconds", to_key="monitoring.interval_seconds"
    ),
    TransformRule(
        from_key="aws.query_frequency_seconds", to_key="aws.query_interval_seconds"
    ),
    TransformRule(
        from_key="aws.ec2.monitoring.cpu.query_frequency_seconds",
        to_key="aws.ec2.monitoring.cpu.query_interval_seconds",
    ),
]
