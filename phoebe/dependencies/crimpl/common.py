from datetime import datetime as _datetime
import os as _os
import sys as _sys
import subprocess as _subprocess
import paramiko as _paramiko
import json as _json
from time import sleep as _sleep

__version__ = '0.2.0'

def _new_job_name():
    return _datetime.now().strftime('%Y.%m.%d-%H.%M.%S')

def _run_cmd(cmd, tunnel=None, detach=False, log_cmd=True, allow_retries=True):
    if cmd is None:
        return
    if log_cmd:
        print("# crimpl{}: {}".format(" (detached)" if detach else "", cmd))

    if tunnel is not None:
        # use the initialized ssh client via paramiko.
        _stdin, _stdout, _stderr = tunnel.exec_command(cmd)
        err = _stderr.read().decode()
        if err:
            raise RuntimeError(err)
        
        cmd_output = _stdout.read().decode().strip()
        return cmd_output

    i = 0
    while True:
        try:
            if detach:
                ret = _subprocess.Popen(cmd, shell=True, stderr=_subprocess.STDOUT)
            else:
                ret = _subprocess.check_output(cmd, shell=True, stderr=_subprocess.STDOUT).decode('utf-8').strip()
        except _subprocess.CalledProcessError as err:
            # print("error output: {}".format(err.output))
            if allow_retries and err.returncode == 255 and i < 5:
                sleeptime = 5+i*5
                print("# crimpl: received ssh error, waiting {}s then retrying".format(sleeptime))
                _sleep(sleeptime)
                i += 1
            else:
                raise
        else:
            if i > 0:
                print("# crimpl: ssh command succeeded")
            return ret

class Server(object):
    def __init__(self, directory=None):
        self._directory = directory
        self._directory_exists = False

        self._dict_keys = ['directory']

        # Initialize an ssh tunnel placeholder; when needed, establish the connection by
        # calling self.connection_init(). When done, call self.connection_close(). That
        # will close the connection and set self.tunnel back to None.
        self.tunnel = None

    def __repr__(self):
        def _format_val(v):
            if isinstance(v, str):
                return "\'{}\'".format(v)
            return v

        return "<{} {}>".format(self.__class__.__name__, " ".join(["{}={}".format(k, _format_val(getattr(self,k))) for k in self._dict_keys]))

    def __str__(self):
        return self.__repr__()

    @property
    def server_name(self):
        """
        internal name of the server.

        Returns
        ----------
        * (string)
        """
        return self._server_name

    @property
    def directory(self):
        if "~" not in self._directory:
            return _os.path.join("~", self._directory)
        return self._directory

    @property
    def existing_jobs(self):
        """
        """
        # TODO: override for EC2 to handle whatever servers are running (if the job server is running, have it check the status, otherwise have the server ec2 check the directory)

        try:
            out = self._run_server_cmd("ls -d {}/crimpl-job-*".format(self.directory))
        except _subprocess.CalledProcessError:
            return []

        directories = out.split()
        job_names = [d.split('crimpl-job-')[-1] for d in directories]
        return job_names

    @property
    def existing_jobs_status(self):
        """
        """
        # TODO: override for EC2 to handle whatever servers are running (if the job server is running, have it check the status, otherwise have the server ec2 check the directory)

        return {job_name: self.get_job(job_name).status for job_name in self.existing_jobs}

    @property
    def conda_installed(self):
        """
        Checks if conda is installed on the remote server

        See also:

        * <<class>.install_conda>

        Returns
        -----------
        * (bool)
        """
        # need to make sure crimpl_directory exists and exportpath.sh is already copied
        self._create_crimpl_directory()
        try:
            out = self._run_server_cmd("conda -V")
        except _subprocess.CalledProcessError:
            return False
        return True

    def venv_exists(self, env_name, env_dir):
        venv_activation = _os.path.join(env_dir, env_name, 'bin/activate')
        try:
            self._run_server_cmd(f'ls {venv_activation}')
        except _subprocess.CalledProcessError:
            return False
        return True

    def _get_conda_envs_dict(self, job_name=None):
        """

        Returns
        ---------
        * (dict)
        """

        # get globally installed environments.  Depending on how conda is
        # setup, the server/job environments MAY not appear here, so we'll
        # check those directories manually later.
        try:
            out = self._run_server_cmd("conda info --envs")
        except _subprocess.CalledProcessError:
            print("# crimpl: conda not yet installed on remote machine")
            return {}


        d = {line.split()[0].split("/")[-1]: line.split()[-1] for line in out.split("\n")[3:] if len(line.split()) > 1}

        # force crimpl environments to override global
        crimpl_env_dir = _os.path.join(self.directory, "crimpl-envs")
        try:
            server_env_paths = self._run_server_cmd("ls -d {}/*".format(crimpl_env_dir))
        except _subprocess.CalledProcessError:
            pass
        else:
            for server_env_path in server_env_paths.split():
                d[server_env_path.split("crimpl-envs/")[-1]] = server_env_path

        # force job clones environments to override crimpl/global
        if job_name is not None:
            crimpl_job_env_dir = _os.path.join(self.directory, "crimpl-job-{}".format(job_name), "crimpl-envs")
            try:
                job_env_paths = self._run_server_cmd("ls -d {}/*".format(crimpl_job_env_dir))
            except _subprocess.CalledProcessError:
                pass
            else:
                for job_env_path in job_env_paths.split():
                    d[job_env_path.split("crimpl-envs/")[-1]] = job_env_path

        return d

    @property
    def conda_envs(self):
        """
        List (existing) available conda environments and their paths on the remote server.

        These will include those created at the root level (either within or outside crimpl)
        as well as those created by this <<class>> (which are stored in <<class>.directory>).
        In the case where the same name is available at the root level and created by
        this <<class>>, the one created by <<class>> will take precedence.

        Returns
        --------
        * (list)
        """
        return list(self._get_conda_envs_dict().keys())

    def connection_init(self):
        """ connection_init() must be implemented in the subclass. """
        pass
    
    def connection_close(self):
        """ connection_init() must be implemented in the subclass. """
        pass

    def _create_crimpl_env(self, crimpl_env, env_name=None, env_dir=None, isolate_env=False, job_name=None, check_if_exists=True, run_cmd=True):
        if crimpl_env == 'none':
            # nothing to be done
            return None, None
        if crimpl_env == 'venv':
            return self._create_venv(env_name, env_dir, check_if_exists=check_if_exists, run_cmd=run_cmd)
        if crimpl_env == 'conda':
            return self._create_conda_env(env_name, isolate_env=isolate_env, job_name=job_name, check_if_exists=check_if_exists, run_cmd=run_cmd)
        raise ValueError(f'crimpl_env={crimpl_env} not supported.')

    def _create_venv(self, env_name, env_dir, check_if_exists=True, run_cmd=True):
        env_path = _os.path.join(env_dir, env_name)

        if check_if_exists and self.venv_exists(env_name, env_dir):
            # venv already exists, nothing to be done.
            return None, env_path

        cmd = f'python -m venv {env_path}'
        
        if run_cmd:
            return self._run_server_cmd(cmd), env_path

        return self.ssh_cmd.format(cmd), env_path

    def _create_conda_env(self, env_name,
                          isolate_env=False,
                          job_name=None,
                          check_if_exists=True,
                          run_cmd=True):
        """
        """
        if env_name is False:
            return None, None

        default_deps = "pip numpy"

        if not (isinstance(env_name, str) or env_name is None):
            raise TypeError("env_name must be a string or None")

        if isinstance(env_name, str) and "/" in env_name:
            raise ValueError("env_name should be alpha-numeric (and -/_) only")

        if env_name is None:
            env_name = 'default'

        python_version = ".".join(_sys.version.split()[0].split(".")[:-1])
        if isolate_env and job_name is not None:
            # need to check to see if the server environment needs to be created and/or cloned
            conda_envs_dict = self._get_conda_envs_dict(job_name=job_name)

            cmd = ""
            envpath_server = _os.path.join(self.directory, "crimpl-envs", env_name)
            envpath = _os.path.join(self.directory, "crimpl-job-{}".format(job_name), "crimpl-envs", env_name)

            if env_name not in conda_envs_dict.keys():
                # create the environment at the server level
                cmd += "conda create -p {envpath_server} -y {default_deps} python={python_version}; ".format(envpath_server=envpath_server, default_deps=default_deps, python_version=python_version)
            if len(cmd) or job_name not in conda_envs_dict.get(env_name):
                # clone the server environment at the job level
                cmd += "conda create -p {envpath} -y --clone {envpath_server};".format(envpath=envpath, envpath_server=envpath_server)

        else:
            if check_if_exists:
                conda_envs_dict = self._get_conda_envs_dict(job_name=job_name)
                if env_name in conda_envs_dict.keys():
                    return None, conda_envs_dict.get(env_name)
            else:
                conda_envs_dict = False

            # create the environment at the server level
            envpath = _os.path.join(self.directory, "crimpl-envs", env_name)
            cmd = "conda create -p {envpath} -y {default_deps} python>={python_version}".format(envpath=envpath, default_deps=default_deps, python_version=python_version)

        if run_cmd:
            return self._run_server_cmd(cmd), envpath
        else:
            return self.ssh_cmd.format(cmd), envpath

    def _create_crimpl_directory(self):
        if self._directory_exists:
            return True

        stdout = _run_cmd(cmd=f'mkdir -p {self.directory}', tunnel=self.tunnel)
        # try:
        #     out = self._run_server_cmd("mkdir -p {directory}".format(directory=self.directory), exportpath=False)
        # except _subprocess.CalledProcessError:
        #     return False
        # else:
        #     # TODO: use temporary file
        #     f = open('exportpath.sh', 'w')
        #     f.write('export PATH="{}/crimpl-bin:$PATH"'.format(self.directory.replace("~", "$HOME")))
        #     f.close()
        #     scp_cmd = self.scp_cmd_to.format(local_path='exportpath.sh', server_path=self.directory+"/")
        #     _run_cmd(scp_cmd)

        self._directory_exists = True
        return True

    def install_conda(self, in_server_directory=False):
        """
        Install conda on the remote server if it is not already installed.

        See also:

        * <<class>.conda_installed>

        Arguments
        -------------
        * `in_server_directory` (bool, optional, default=False): whether to place
            the conda installation in <<class>.directory> rather than the default
            user installation

        Returns
        ------------
        * (bool): output of <<class>.conda_installed>
        """
        if self.conda_installed:
            return

        if in_server_directory:
            out = self._run_server_cmd("cd {directory}; wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; sh Miniconda3-latest-Linux-x86_64.sh -u -b -p ./crimpl-conda; mkdir ./crimpl-bin; cp ./crimpl-conda/bin/conda ./crimpl-bin/conda".format(directory=self.directory, exportpath=False))
        else:
            out = self._run_server_cmd("cd {directory}; wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh; sh Miniconda3-latest-Linux-x86_64.sh -u -b; mkdir ./crimpl-bin; cp ~/miniconda3/bin/conda ./crimpl-bin".format(directory=self.directory, exportpath=False))
        out = self._run_server_cmd("conda init")

        return self.conda_installed

    def _submit_script_cmds(self, script, files, ignore_files,
                            use_scheduler,
                            directory,
                            crimpl_env, env_name, env_dir,
                            isolate_env,
                            job_name,
                            terminate_on_complete=False,
                            use_nohup=False,
                            install_conda=False,
                            **sched_kwargs):

        if crimpl_env != 'conda' and isolate_env is True:
            raise ValueError(f'cannot use isolate_env with crimpl_env={crimpl_env}.')
        # from job: self.server._submit_script_cmds(script, files, use_slurm, directory=self.remote_directory, isolate_env=self.isolate_env, job_name=self.job_name)
        # from server: self._submit_script_cmds(script, files, use_slurm=False, directory=self.directory, isolate_env=False, job_name=None)

        # NOTE: job_name here is used to identify IF a job and as the slurm job name, but is NOT necessary the job.job_name
        if isinstance(script, str):
            # TODO: allow for http?
            if not _os.path.isfile(script):
                raise ValueError("cannot find valid script at {}.  To pass commands directly, pass a list of strings".format(script))

            f = open(script, 'r')
            script = script.readlines()

        if not isinstance(script, list):
            raise TypeError("script must be of type string (path) or list (list of commands)")

        # for i, line in enumerate(script):
        #     # ensure that all calls to conda install without prompts?
        #     if "conda" in line and "-y" not in line:
        #         script[i] = script[i] + " -y"

        if crimpl_env == 'conda':
            if install_conda:
                self.install_conda(in_server_directory=True)
            elif not self.conda_installed:
                raise ValueError("conda is not installed on the remote server. Please install manually or call server.install_conda().")

        def _slurm_kwarg_to_prefix(k):
            exceptions = {'nprocs': '-n ',
                          'walltime': '-t ',
                          'mail_type': '--mail-type=',
                          'mail_user': '--mail-user='}
            if k in exceptions.keys():
                return exceptions.get(k)
            elif len(k) == 1:
                return f"-{k} "
            else:
                return f"--{k}="

        create_env_cmd, env_path = self._create_crimpl_env(
            crimpl_env, env_name=env_name, env_dir=env_dir,
            isolate_env=isolate_env, job_name=job_name, check_if_exists=True,
            run_cmd=False)

        if use_scheduler and job_name is None:
            raise ValueError("use_scheduler requires job_name")
        if use_scheduler and use_nohup:
            raise ValueError("cannot use both use_scheduler and use_nohup")

        if job_name is not None:
            if use_scheduler:
                sched_script = ["#!/bin/bash"]

                if use_scheduler == 'slurm':
                    sched_script += ["#SBATCH -D {}".format(directory+"/")]
                    sched_script += ["#SBATCH -J {}".format(job_name)]

                    for k, v in sched_kwargs.items():
                        if v is None:
                            continue
                        prefix = _slurm_kwarg_to_prefix(k)
                        if prefix is False:
                            raise NotImplementedError("slurm command for {} not implemented".format(k))
                        if k == 'mail_type' and isinstance(v, list):
                            v = ",".join(v)
                        sched_script += ["#SBATCH {}{}".format(prefix, v)]

                    if crimpl_env == 'venv':
                        script += [f'source {_os.path.join(env_path, "bin/activate")}']
                else:
                    raise NotImplementedError("use_scheduler={} not implemented".format(use_scheduler))

                orig_script = script
                script = sched_script + ["\n\n", "echo \'starting\' > crimpl-job.status"]
                if crimpl_env == 'conda':
                    script += ["eval \"$(conda shell.bash hook)\"", "conda activate {}".format(env_path)]
                script += ["echo \'running\' > crimpl-job.status"] + orig_script + ["echo \'complete\' > crimpl-job.status"]

            else:
                # need to track status by writing to log file
                if "#!" in script[0]:
                    script = [script[0]] + ["echo \'running\' > crimpl-job.status"] + script[1:] + ["echo \'complete\' > crimpl-job.status"]
                else:
                    script = ["echo \'running\' > crimpl-job.status"] + script + ["echo \'complete\' > crimpl-job.status"]


        # TODO: use tmp file instead
        script_fname = 'crimpl_submit_script.sh' if use_scheduler or use_nohup else 'crimpl_run_script.sh'
        with open(script_fname, 'w') as f:
            if not use_scheduler:
                f.write("echo \'starting\' > crimpl-job.status\n")
                if crimpl_env == 'conda':
                    f.write("eval \"$(conda shell.bash hook)\"\nconda activate {}\n".format(env_path))
                if crimpl_env == 'venv':
                    f.write(f'source {_os.path.join(env_path, "bin/activate")}\n')
            f.write("\n".join(script))

            if terminate_on_complete:
                # should really only be used for future AWS support
                f.write("\nsudo shutdown now")

        if not isinstance(files, list):
            raise TypeError("files must be of type list")
        for f in files:
            if not _os.path.isfile(f):
                raise ValueError("cannot find file at {}".format(f))

        mkdir_cmd = self.ssh_cmd.format("mkdir -p {}".format(directory))
        if job_name is not None:
            logfiles_cmd = self.ssh_cmd.format("echo \'{}\' >> {}".format(" ".join([_os.path.basename(f) for f in files+ignore_files]), _os.path.join(directory, "crimpl-input-files.list"))) if len(files+ignore_files) else self.ssh_cmd.format("touch {}".format(_os.path.join(directory, "crimpl-input-files.list")))
            logenv_cmd = self.ssh_cmd.format("echo \'{}\' > {}".format(env_name, _os.path.join(directory, "crimpl-conda-environment")))

        # TODO: use job subdirectory for server_path
        scp_cmd = self.scp_cmd_to.format(local_path=" ".join([script_fname]+[_os.path.normpath(f).replace(' ', '\ ') for f in files]), server_path=directory+"/")

        if use_scheduler:
            if use_scheduler == 'slurm':
                remote_script = _os.path.join(directory, _os.path.basename(script_fname))
                cmd = self.ssh_cmd.format("sbatch {remote_script}".format(remote_script=remote_script))
        else:
            remote_script = "./"+script_fname
            if use_nohup:
                cmd = self.ssh_cmd.format(f"cd {directory}; chmod +x {remote_script}; nohup bash {remote_script} 2> {remote_script}.err & echo $! > crimpl-nohup.pid")
            else:
                cmd = self.ssh_cmd.format(f"cd {directory}; chmod +x {remote_script}; {remote_script} 2> {remote_script}.err")
        if job_name is not None:
            return [mkdir_cmd, scp_cmd, logfiles_cmd, logenv_cmd, create_env_cmd, cmd]
        else:
            return [mkdir_cmd, scp_cmd, create_env_cmd, cmd]

    def create_job(self, job_name=None, crimpl_env='none', env_name=None, env_dir=None, isolate_env=False):
        """
        """
        return self._JobClass(server=self,
                              job_name=job_name,
                              crimpl_env=crimpl_env,
                              env_name=env_name,
                              env_dir=env_dir,
                              isolate_env=isolate_env,
                              connect_to_existing=False)

    def get_job(self, job_name=None):
        """
        """
        return self._JobClass(server=self, job_name=job_name, connect_to_existing=True)

    def submit_job(self, *args, **kwargs):
        raise NotImplementedError("submit_job not subclassed by {}".format(self.__class__.__name__))

    def to_dict(self):
        """
        Dictionary representation of the server configuration.

        Returns
        ----------
        * (dict)
        """
        d = {k: getattr(self, k) if hasattr(self, k) else getattr(self, "_{}".format(k)) for k in self._dict_keys}
        d['crimpl'] = self.__class__.__name__
        d['crimpl.version'] = __version__
        return d

    def save(self, name=None, overwrite=False):
        """
        Save this server configuration to ~/.crimpl to be loaded again via
        <crimpl.load_server>.

        Note that this saves everything in <<class>.to_dict> to disk in ASCII.

        Arguments
        ----------
        * `name` (string, optional, default=None): name of the server.  Will
            default to <<class>.server_name> if set.
        * `overwrite` (bool, optional, default=False): whether to overwrite
            an existing saved configuration for `name`.

        Returns
        ----------
        * (string): path to the saved ascii file

        Raises
        ----------
        * ValueError: if `name` is already saved but `overwrite` is not passed as True
        """
        if name is None:
            name = self.server_name

        if name is None:
            raise ValueError("must pass name or set server_name")

        directory = _os.path.expanduser("~/.crimpl/servers")
        if not _os.path.exists(directory):
            _os.makedirs(directory)

        fname = _os.path.join(directory, "{}.json".format(name))
        if _os.path.exists(fname) and not overwrite:
            raise ValueError("server with name='{}' already exists at {}.  Use a different name or pass overwrite=True".format(name, fname))
        with open(fname, 'w') as f:
            json = self.to_dict()
            _json.dump(json, f)

        return fname


class ServerJob(object):
    def __init__(self, server, job_name=None,
                 crimpl_env='none', env_dir='~/.venvs', env_name='phoebe',
                 isolate_env=False,
                 job_submitted=False):
        self._server = server

        self._job_name = job_name
        self._job_submitted = job_submitted

        crimpl_env_options = ['none', 'conda', 'venv']
        if crimpl_env not in crimpl_env_options:
            raise ValueError(f'crimpl_env={crimpl_env} is not in one of the supported options ({crimpl_env_options})')
        self._crimpl_env = crimpl_env

        if not isinstance(env_dir, str):
            raise ValueError(f'env_dir should be a string (a path to the venv environment).')
        self._env_dir = env_dir

        if not isinstance(env_name, str) or (isinstance(env_name, str) and '/' in env_name):
            raise ValueError(f'env_name should be a string that does not contain "/".')
        self._env_name = env_name

        if not isinstance(isolate_env, bool):
            raise TypeError("isolate_env must be of type bool")
        self._isolate_env = isolate_env

        # allow caching once the environment exists
        self._crimpl_env_exists = False

        # allow for caching remote_directory
        self._remote_directory = None

        # allow caching for input files
        self._input_files = None

    def __repr__(self):
        return "<{} job_name=\'{}\'>".format(self.__class__.__name__, self.job_name)

    def __str__(self):
        return self.__repr__()

    @property
    def server(self):
        """
        Access the parent server object
        """
        return self._server

    @property
    def crimpl_env(self):
        """
        Type of the crimpl virtual environment.
        """
        return self._crimpl_env

    @property
    def env_dir(self):
        """
        Directory that hosts a venv virtual environment.
        """
        return self._env_dir

    @property
    def env_name(self):
        """
        Name of the remote virtual environment (conda or venv).
        """
        return self._env_name

    @property
    def isolate_env(self):
        """
        """
        return self._isolate_env

    @property
    def crimpl_env_exists(self):
        if self._crimpl_env == 'none':
            return True

        if self._crimpl_env_exists:
            return True
        
        if self._crimpl_env == 'conda':
            self._crimpl_env_exists = self.env_name in self.server.conda_envs

        if self._crimpl_env == 'venv':
            self._crimpl_env_exists = self.venv_exists(self.env_name, self.env_dir)

        return self._crimpl_env_exists

    def create_crimpl_env(self):
        """
        """
        
        if self.crimpl_env_exists:
            return

        return self.server._create_crimpl_env(check_if_exists=False)

    def create_conda_env(self):
        """
        Create a conda environment `self.env_name` in the <<Server>.remote_directory>).

        This environment will be available to any jobs in this server and will
        be listed in <<Server>.conda_envs>.  The created environment will
        use the same version of python as the local version.
        """
        if self.crimpl_env_exists:
            return

        return self.server._create_crimpl_env(self.env_name, check_if_exists=False)

    @property
    def job_name(self):
        """
        Access the job name

        Returns
        ----------
        * (string)
        """
        return self._job_name

    @property
    def remote_directory(self):
        """
        Access the **job** subdirectory location on the remote server.

        Returns
        ----------
        * (string)
        """
        if self._remote_directory is None:
            home_dir = self.server._run_server_cmd("pwd")
            if "~" in self.server.directory:
                self._remote_directory = _os.path.join(self.server.directory.replace("~", home_dir), "crimpl-job-{}".format(self.job_name))
            else:
                self._remote_directory = _os.path.join(home_dir, self.server.directory, "crimpl-job-{}".format(self.job_name))
        return self._remote_directory

    @property
    def ls(self):
        """
        List all files in the **job** subdirectory on the remote server.

        See also:

        * <<class>.job_files>
        * <<class>.input_files>
        * <<class>.output_files>

        Returns
        ----------
        * (list)
        """
        try:
            response = self.server._run_server_cmd("ls {}/*".format(self.remote_directory))
        except _subprocess.CalledProcessError:
            return []
        return [_os.path.basename(f) for f in response.split()]

    @property
    def job_status(self):
        """
        Return the status of the job by checking the logged status file in the remote directory.

        Returns
        -----------
        * (string): one of not-submitted, pending, running, canceled, failed, complete, unknown
        """
        if not self._job_submitted:
            return 'not-submitted'

        return self.server._run_server_cmd("cat {}".format(_os.path.join(self.remote_directory, "crimpl-job.status")))

    @property
    def job_files(self):
        """
        List the files in the **job** subdirectory on the remote server, including
        files sent to the server and output files, but excluding files created
        and managed by **crimpl**.

        See also:

        * <<class>.ls>
        * <<class>.input_files>
        * <<class>.output_files>

        Returns
        -------------
        * (list)
        """
        return [f for f in self.ls if f[:6]!='crimpl']

    @property
    def input_files(self):
        """
        List the **input** files in the **job** subdirectory on the remote server.

        These were files sent via <<class>.submit_job>.

        See also:

        * <<class>.ls>
        * <<class>.job_files>
        * <<class>.output_files>

        Returns
        -----------
        * (list)
        """
        if self._input_files is None:

            response = self.server._run_server_cmd("cat {}".format(_os.path.join(self.remote_directory, "crimpl-input-files.list")))
            self._input_files = response.split()

        return self._input_files

    @property
    def output_files(self):
        """
        List the **output** files in the **job** subdirectory on the remote server.

        These are all <<class>.job_files> that are not included in <<class>.input_files>.

        See also:

        * <<class>.ls>
        * <<class>.job_files>
        * <<class>.input_files>

        Returns
        ----------
        * (list)
        """
        return [f for f in self.job_files if f not in self.input_files]

    def wait_for_job_status(self, status='complete',
                            error_if=['failed', 'canceled'],
                            sleeptime=5):
        """
        Wait for the job to reach a desired job_status.

        Arguments
        -----------
        * `status` (string or list, optional, default='complete'): status
            or statuses to exit the wait loop successfully.
        * `error_if` (string or list, optional, default=['failed', 'canceled']): status or
            statuses to exit the wait loop and raise an error.
        * `sleeptime` (int, optional, default=5): number of seconds to wait
            between successive job status checks.

        Returns
        ----------
        * (string): `status`
        """
        if status is True:
            status = 'complete'

        if isinstance(status, str):
            status = [status]

        if isinstance(error_if, str):
            error_if = [error_if]

        while True:
            job_status = self.job_status
            print("# crimpl: job_status={}".format(job_status))
            if job_status in status:
                break
            if job_status in error_if:
                raise ValueError("job_status={}".format(job_status))
            _sleep(sleeptime)

        return job_status

    def check_output(self, server_path=None, local_path="./"):
        """
        Attempt to copy a file(s) back from the remote server.

        Arguments
        -----------
        * `server_path` (string or list or None, optional, default=None): path(s)
            (relative to `directory`) on the server of the file(s) to retrieve.
            If not provided or None, will default to <<class>.output_files>.
            See also: <<class>.ls> or <<class>.job_files> for a full list of
            available files on the remote server.
        * `local_path` (string, optional, default="./"): local path to copy
            the retrieved file.


        Returns
        ----------
        * (list) list of retrieved files
        """
        if isinstance(server_path, str):
            server_path = [server_path]

        if server_path is None:
            server_path = self.output_files
        else:
            server_path = [p for p in server_path if p in self.ls]

        if isinstance(server_path, list):
            if not len(server_path):
                return []
            if len(server_path) == 1:
                server_path = server_path[0]

        if self.server.scp_cmd_from[:3] != "scp" and isinstance(server_path, list):
            # cp doesn't like the {} formatting
            for path in server_path:
                cp_cmd = self.server.scp_cmd_from.format(server_path=_os.path.join(self.remote_directory, path), local_path=local_path)
                _run_cmd(cp_cmd)
            return [server_path] if isinstance(server_path, str) else server_path

        if isinstance(server_path, str):
            server_path_str = _os.path.join(self.remote_directory, server_path)
        else:
            server_path_str = "%s/{%s}" %  (self.remote_directory, ",".join(server_path))

        scp_cmd = self.server.scp_cmd_from.format(server_path=server_path_str, local_path=local_path)
        # TODO: execute cmd, and handle errors if stopped/terminated before getting results
        _run_cmd(scp_cmd)

        return [server_path] if isinstance(server_path, str) else server_path



class SSHServer(Server):

    @property
    def _ssh_cmd(self):
        raise NotImplementedError("{} does not subclass _ssh_cmd".format(self.__class__.__name__))

    @property
    def ssh_cmd(self):
        """
        ssh command to the server

        Returns
        ----------
        * (string): command with "{}" placeholders for the command to run on the remote machine.
        """
        # return "%s \'export PATH=\"%s/crimpl-bin:$PATH\"; {}\'" % (self._ssh_cmd, self.directory.replace("~", "$HOME"))
        return "%s \"source %s/exportpath.sh; {}\"" % (self._ssh_cmd, self.directory)

        # TODO: need to create a directory/exportpath.sh EXECUTABLE file that does the same as above

    def _run_server_cmd(self, cmd, exportpath=None, allow_retries=True):
        if cmd is None:
            return

        if cmd[:3] == 'scp':
            ssh_cmd = cmd
        elif cmd[:3] == 'ssh':
            ssh_cmd = cmd
        else:
            if exportpath is None:
                exportpath = 'conda' in cmd or 'crimpl_submit_script.sh' in cmd or 'crimpl_run_script.sh' in cmd

            if exportpath:
                ssh_cmd = self.ssh_cmd.format(cmd)
            else:
                ssh_cmd = "{} \"{}\"".format(self._ssh_cmd, cmd)
            # ssh_cmd = self.ssh_cmd+" \'export PATH=\"{directory}/crimpl-bin:$PATH\"; {cmd}\'".format(directory=self.directory.replace("~", "$HOME"), cmd=cmd)
            # ssh_cmd = self.ssh_cmd+" \'{cmd}\'".format(directory=self.directory.replace("~", "$HOME"), cmd=cmd)

        try:
            return _run_cmd(ssh_cmd, allow_retries=allow_retries)
        except _subprocess.CalledProcessError as err:
            if "2>" in ssh_cmd:
                error_file = _os.path.join(self.directory, ssh_cmd.split("2>")[1].split()[0].split('\"')[0])
                print("# crimpl: received error when running command, expecting stderr to be written to {}".format(error_file))
                error_msg = self._run_server_cmd("cat {}".format(error_file), exportpath=False, allow_retries=True)
                raise ValueError("server raised error: {}".format(error_msg))
            else:
                raise

    @property
    def scp_cmd_to(self):
        raise NotImplementedError("{} does not subclass scp_cmd_to".format(self.__class__.__name__))

    @property
    def scp_cmd_from(self):
        raise NotImplementedError("{} does not subclass scp_cmd_from".format(self.__class__.__name__))
