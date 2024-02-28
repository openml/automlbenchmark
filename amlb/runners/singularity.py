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

from ..benchmark import _setup_dir_
from ..resources import config as rconfig, get as rget
from ..utils import dir_of, run_cmd, touch
from .container import ContainerBenchmark


log = logging.getLogger(__name__)


class SingularityBenchmark(ContainerBenchmark):
    """SingularityBenchmark
    an extension of ContainerBenchmark to run benchmarks inside Singularity.
    """

    @classmethod
    def image_name(cls, framework_def, label=None, as_docker_image=False, **kwargs):
        """
        We prefer to pull from docker, so we have to mind the docker tag
        When downloading from Docker, the colon is changed to underscore
        """
        if label is None:
            label = rget().project_info.branch
        di = framework_def.image

        # If we want to pull from docker, the separator is a colon for tag
        separator = '_' if not as_docker_image else ':'
        # Also, no need for author in image name
        author = '' if not as_docker_image else f"{di.author}/"
        image = di.image if di.image else framework_def.name.lower()
        tags = [di.tag if di.tag else framework_def.version.lower()]
        if label not in rconfig().container.ignore_labels:
            tags.append(label)
        tag = re.sub(r"([^\w.-])", '.', '-'.join(tags))
        return f"{author}{image}{separator}{tag}"

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

    def _container_image_name(self, label=None, as_docker_image=False):
        """
        Singularity Images would be located on the framework directory
        """
        image_name = self.image_name(self.framework_def, label=label, as_docker_image=as_docker_image)

        # Make sure image is in the framework directory
        if as_docker_image:
            return image_name
        else:
            return os.path.join(self._framework_dir, _setup_dir_, image_name + '.sif')

    @property
    def _script(self):
        return os.path.join(self._framework_dir, _setup_dir_, 'Singularityfile')

    def _start_container(self, script_params=""):
        """Implementes the container run method"""
        in_dir = rconfig().input_dir
        out_dir = rconfig().output_dir
        custom_dir = rconfig().user_dir
        for d in [in_dir, out_dir, custom_dir]:
            touch(d, as_dir=True)
        script_extra_params = "--session="  # in combination with `self.output_dirs.session` usage below to prevent creation of 2 sessions locally
        cmd = (
            "singularity run --pwd /bench {options} "
            "-B {input}:/input -B {output}:/output -B {custom}:/custom "
            "{image} \"{params} -i /input -o /output -u /custom -s skip -Xrun_mode=singularity {extra_params}\""
        ).format(
            options=rconfig().singularity.run_extra_options,
            input=in_dir,
            output=self.output_dirs.session,
            custom=custom_dir,
            image=self.image,
            params=script_params,
            extra_params=script_extra_params,
        )
        log.info("Starting Singularity: %s.", cmd)
        log.info("Datasets are loaded by default from folder %s.", in_dir)
        log.info("Generated files will be available in folder %s.", out_dir)
        try:
            run_cmd(cmd, _capture_error_=False)  # console logs are written on stderr by default: not capturing allows live display
        except:
            # also want to handle KeyboardInterrupt
            # In the foreground run mode, the user has to kill the process
            # There is yet no docker kill command. User has to kill PID manually
            log.warning(f"Container {inst_name} may still be running, please verify and kill it manually.")
            raise Exception

    def _image_exists(self, image):
        """Implements a method to see if the container image is available"""
        log.info(f"Looking for the image {image}")
        if os.path.exists(image):
            return True
        try:
            # We pull from docker as there are not yet singularity org accounts
            run_cmd("singularity pull {output_file} docker://{image}".format(
                image=self._container_image_name(as_docker_image=True),
                output_file=image,
            ), _live_output_=True)
            return True
        except Exception:
            try:
                # If no docker image, pull from singularity hub
                run_cmd("singularity pull {output_file} library://{library}/{image}".format(
                    image=self._container_image_name(as_docker_image=True),
                    output_file=image,
                    library=rconfig().singularity.library
                ), _live_output_=True)
                return True
            except Exception:
                pass
        return False

    def _run_container_build_command(self, image, cache):
        log.info(f"Building singularity image {image}.")
        run_cmd("sudo singularity build {options} {container} {script}".format(
            options="" if cache else "--disable-cache",
            container=image,
            script=self._script,
        ), _live_output_=True)
        log.info(f"Successfully built singularity image {image}.")

    def _upload_image(self, image):
        library = rconfig().singularity.library
        name = self._container_image_name(as_docker_image=True)
        log.info(f"Publishing Singularity image {image}.")
        run_cmd(f"singularity login && singularity push -U {image} library://{library}/{name}")
        log.info(f"Successfully published singularity image {image}.")

    def _generate_script(self, custom_commands):
        singularity_content = """Bootstrap: docker
From: ubuntu:22.04
%files
. /bench/
%post

DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get -y install apt-utils dialog locales
apt-get -y install curl wget unzip git
apt-get -y install software-properties-common
add-apt-repository -y ppa:deadsnakes/ppa
apt-get update
apt-get -y install python{pyv} python{pyv}-venv python{pyv}-dev python3-pip
#update-alternatives --install /usr/bin/python3 python3 $(which python{pyv}) 1

# aliases for the python system
SPIP="python{pyv} -m pip"
SPY=python{pyv}

# Enforce UTF-8 encoding
PYTHONUTF8=1
PYTHONIOENCODING=utf-8
# RUN locale-gen en-US.UTF-8
LANG=C.UTF-8
LC_ALL=C.UTF-8

cd /bench

# We create a virtual environment so that AutoML systems may use their preferred versions of
# packages that we need to data pre- and postprocessing without breaking it.
$SPIP install -U pip wheel
$SPY -m venv venv
PIP="/bench/venv/bin/python{pyv} -m pip"
PY="/bench/venv/bin/python{pyv} -W ignore"
#$PIP install -U pip=={pipv} wheel
$PIP install -U pip wheel

mkdir /input
mkdir /output
mkdir /custom


(grep -v '^\\s*#' | xargs -L 1 $PIP install --no-cache-dir) < requirements.txt

$PY {script} {framework} -s only
{custom_commands}

%environment
export DEBIAN_FRONTEND=noninteractive
export SPIP=python{pyv} -m pip
export SPY=python{pyv}
export PYTHONUTF8=1
export PYTHONIOENCODING=utf-8
export LANG=C.UTF-8
export LC_ALL=C.UTF-8
export PIP=/bench/venv/bin/python{pyv} -m pip
export PY=/bench/venv/bin/python{pyv}
%runscript
cd /bench
exec /bin/bash -c "$PY {script} ""$@"
%startscript
cd /bench
exec /bin/bash -c "$PY {script} ""$@"

""".format(
            custom_commands=custom_commands.format(
                setup=dir_of(
                    os.path.join(self._framework_dir, "setup/"),
                    rel_to_project_root=True
                ),
                pip="$PIP",
                py="$PY"
            ),
            framework=self._forward_params['framework_name'],
            pyv=rconfig().versions.python,
            pipv=rconfig().versions.pip,
            script=rconfig().script,
        )

        touch(self._script)
        with open(self._script, 'w') as file:
            file.write(singularity_content)
