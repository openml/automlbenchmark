"""
**Singularity** module is build on top of **ContainerBenchmark** module to provide logic to create and run singularity images
that are preconfigured with a given automl framework, and that can be used to run a benchmark anywhere.
The Singularity image embeds a version of the automlbenchmark app so that tasks are later run in local mode inside singularity,
providing the same parameters and features allowing to import config and export results through mounted folders.
The image is pulled form an existing docker, yet executed in singularity framework
"""
import logging
import os
import re

from .container import ContainerBenchmark
from .resources import config as rconfig, get as rget
from .utils import dir_of, run_cmd


log = logging.getLogger(__name__)


class SingularityBenchmark(ContainerBenchmark):
    """SingularityBenchmark
    an extension of ContainerBenchmark to run benchmarks inside docker.
    """

    def __init__(self, framework_name, benchmark_name, constraint_name):
        """

        :param framework_name:
        :param benchmark_name:
        :param constraint_name:
        """
        super().__init__(framework_name, benchmark_name, constraint_name)
        self._custom_image_name = rconfig().singularity.image
        self.minimize_instances = rconfig().singularity.minimize_instances
        self.container_name = 'singularity'
        self.force_branch = rconfig().singularity.force_branch
        self.custom_commands = self.framework_module.singularity_commands(
            self.framework_def.setup_args,
            setup_cmd=self.framework_def._setup_cmd
        ) if hasattr(self.framework_module, 'singularity_commands') else ""

    @property
    def _script(self):
        return os.path.join(self._framework_dir, 'Singularityfile')

    def _start(self, script_params=""):
        """Implementes the container run method"""
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        script_extra_params = ""
        inst_name = self.sid
        cmd = (
            "singularity run {options} "
            "-B {input}:/input -B {output}:/output -B {custom}:/custom "
            "{image} \"{params} -i /input -o /output -u /custom -s skip -Xrun_mode=singularity {extra_params}\""
        ).format(
            name=inst_name,
            options=rconfig().singularity.run_extra_options,
            input=in_dir,
            output=out_dir,
            custom=custom_dir,
            image=self._docker2singularity(self._image_name),
            params=script_params,
            extra_params=script_extra_params,
        )
        log.info("Starting Singularity: %s.", cmd)
        log.info("Datasets are loaded by default from folder %s.", in_dir)
        log.info("Generated files will be available in folder %s.", out_dir)
        try:
            run_cmd(cmd, _capture_error_=False)  # console logs are written on stderr by default: not capturing allows live display
        except:  # also want to handle KeyboardInterrupt
            try:
                raise NotImplementedError
            except:
                pass
            finally:
                raise
    def _docker2singularity(self, name):
        author, image_name, tag = re.split('/|:', name)
        return image_name + '_' + tag + '.sif'

    def _image_exists(self):
        """Implements a method to see if the container image is available"""
        log.info(f"Looking for the image {self._image_name}")
        if os.path.exists(self._docker2singularity(self._image_name)):
            return True
        try:
            # We pull from docker as there are not yet singularity org accounts
            run_cmd("singularity pull docker://{image}".format(image=self._image_name))
            return True
        except:
            pass
        return False

    def _run_container_build_command(self, cache):
        log.info(f"Building singularity image {self._image_name}.")
        name = self._docker2singularity(self._image_name)
        run_cmd("sudo singularity build {options} {container} {script}".format(
            options="" if cache else "--disable-cache",
            container=self._docker2singularity(self._image_name),
            script=self._script,
        ), _live_output_=True)
        log.info(f"Successfully built singularity image {name}.")

    def _upload_image(self):
        image = self._image_name
        library=rconfig().singularity.library
        name = self._docker2singularity(self._image_name)
        log.info(f"Publishing Singularity image {image}.")
        run_cmd(f"singularity login && singularity push -U {name} library://{library}/{name}:latest")
        log.info(f"Successfully published singularity image {image}.")


    def _generate_script(self, custom_commands):
        singularity_content="""Bootstrap: docker
From: ubuntu:18.04
%files
. /bench/
%post

DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y install apt-utils dialog locales
apt-get -y install curl wget unzip git
apt-get -y install python3 python3-pip python3-venv
pip3 install -U pip

# aliases for the python system
SPIP=pip3
SPY=python3

# Enforce UTF-8 encoding
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
# RUN locale-gen en-US.UTF-8
LANG=C.UTF-8
LC_ALL=C.UTF-8

cd /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
$SPY -m venv venv
PIP=/bench/venv/bin/pip3
PY=/bench/venv/bin/python3
#RUN $PIP install -U pip=={pip_version}
$PIP install -U pip

mkdir /input
mkdir /output
mkdir /custom

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)

xargs -L 1 $PIP install --no-cache-dir < requirements.txt

{custom_commands}

%environment
export DEBIAN_FRONTEND=noninteractive
export SPIP=pip3
export SPY=python3
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PIP=/bench/venv/bin/pip3
export PY=/bench/venv/bin/python3
%runscript
cd /bench
exec /bin/bash -c "$PY {script} ""$@"
%startscript
cd /bench
exec /bin/bash -c "$PY {script} ""$@"

""".format(
            custom_commands=custom_commands.format(
                **dict(
                    setup=dir_of(
                        os.path.join(self._framework_dir, "setup/"),
                        rel_to_project_root=True
                    ),
                    pip="$PIP",
                    py="$PY"
                )
            ),
            framework=self.framework_name,
            pip_version=rconfig().versions.pip,
            script=rconfig().script,
            user=rconfig().user_dir,
        )

        with open(self._script, 'w') as file:
            file.write(singularity_content)
