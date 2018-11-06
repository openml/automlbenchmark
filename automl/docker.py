import os

from .benchmark import Benchmark


class DockerBenchmark(Benchmark):
    """DockerBenchmark
    an extension of Benchmark to run benchmarks inside docker.
    """

    def __init__(self, framework_name, benchmark_name, config, reuse_instance=False):
        """

        :param framework_name:
        :param benchmark_name:
        :param config:
        :param reuse_instance:
        """
        super().__init__(framework_name, benchmark_name, config)
        self.reuse_instance = reuse_instance

    def setup(self):
        custom_commands = self.framework_module.docker_commands() if hasattr(self.framework_module, 'docker_commands') else ""
        self._generate_docker_script(custom_commands)
        self._build_docker_image()

    def run(self):
        if self.reuse_instance:
            self.start_docker("{framework} {benchmark}".format(
                framework=self.framework_def.name,
                benchmark=self.benchmark_name
            ), verbose=True)
        else:
            super().run()

    def _run_fold(self, task_def, fold: int):
        self.run_one(task_def.name, fold)

    def run_one(self, task_name: str, fold: int):
        self.start_docker("{framework} {benchmark} -t {task} -f {fold}".format(
            framework=self.framework_def.name,
            benchmark=self.benchmark_name,
            task=task_name,
            fold=fold
        ), verbose=True)

    def start_docker(self, script_params="", verbose=False):
        cmd = "docker run -v {output}:/output --rm {image} {params} -o /output".format(
            output=self.resources.config['output_folder'],
            image=self._docker_image_name,
            params=script_params
        )
        print("starting docker: " + cmd)
        output = os.popen(cmd).read()
        if verbose:
            print(output)

    @property
    def _docker_script(self):
        return os.path.join(self._framework_dir, 'Dockerfile')

    def _generate_docker_script(self, custom_commands):
        docker_content = """FROM ubuntu:18.04

RUN apt-get update
RUN apt-get install -y curl wget unzip git
RUN apt-get install -y python3 python3-pip python3-venv
RUN pip3 install --upgrade pip

# We create a virtual environment so that AutoML systems may use their preferred versions of 
# packages that we need to data pre- and postprocessing without breaking it.
ENV V_PIP /venvs/setup/bin/pip3
ENV V_PY /venvs/setup/bin/python3
ENV PIP pip3
ENV PY python3

RUN $PY -m venv /venvs/setup
RUN $V_PIP install --upgrade pip

WORKDIR /bench
VOLUME /output

# Add the AutoML system except files listed in .dockerignore (could also use git clone directly?)
ADD . /bench/

RUN $PIP install --no-cache-dir -r requirements.txt --process-dependency-links
RUN $PIP install --no-cache-dir openml

{custom_commands}

# https://docs.docker.com/engine/reference/builder/#entrypoint
ENTRYPOINT ["/bin/bash", "-c", "$PY runbenchmark.py $0 $*"]
CMD ["constantpredictor", "test"]

""".format(custom_commands=custom_commands)
        with open(self._docker_script, 'w') as file:
            file.write(docker_content)

    @property
    def _docker_image_name(self):
        docker_image = self.framework_def.docker_image
        return "{author}/{image}:{tag}".format(
            author=docker_image["author"],
            image=docker_image["image"],
            tag=docker_image["tag"]
        )

    def _build_docker_image(self, verbose=False):
        output = os.popen("docker build -t {container} -f {script} .".format(
            container=self._docker_image_name,
            script=self._docker_script
        )).read()
        if verbose:
            print(output)
