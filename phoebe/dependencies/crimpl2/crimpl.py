import os
import paramiko as pm
import json
import base64
from datetime import datetime as dt


__version__ = '0.3.0'


class CrimplServer(object):
    def __init__(self, host='localhost', working_dir='~/.crimpl'):
        self.host = host
        self.working_dir = working_dir
        self.tunnel = None

    def __str__(self):
        return f'<CrimplServer host={self.host}>'

    def __repr__(self):
        return f'<CrimplServer host={self.host}>'
    
    def connect(self):
        """ Should be overloaded by the subclass. """
        pass

    def disconnect(self):
        """ Should be overloaded by the subclass. """
        pass


class SlurmServer(CrimplServer):
    def __init__(self, host='localhost', working_dir='~/.crimpl', username=None, password=None, pkey=None, crimpl_env='none', env_name=None, env_dir=None, extra_path=None, extra_ld_library_path=None):
        self.host = host
        self.working_dir = working_dir
        self.username = username
        self.password = password
        self.pkey = pkey

        self.crimpl_env = crimpl_env
        self.env_name = env_name
        self.env_dir = env_dir

        self.venv = os.path.join(env_dir, env_name) if crimpl_env == 'venv' else None
        self.conda = env_name if crimpl_env == 'conda' else None

        self.extra_path = extra_path
        self.extra_ld_library_path = extra_ld_library_path

        self.tunnel = None
    
    def __str__(self):
        return f'<SlurmServer host={self.host}>'

    def __repr__(self):
        return f'<SlurmServer host={self.host}>'

    def save(self, name=None, overwrite=False):
        if name is None:
            name = self.host

        directory = os.path.expanduser("~/.crimpl2/servers")
        if not os.path.exists(directory):
            os.makedirs(directory)

        fname = os.path.join(directory, "{}.json".format(name))
        if os.path.exists(fname) and not overwrite:
            raise ValueError("server with name='{}' already exists at {}.  Use a different name or pass overwrite=True".format(name, fname))

        config = dict()
        with open(fname, 'w') as conffile:
            config['crimpl'] = self.__class__.__name__
            config['crimpl.version'] = __version__
            config['host'] = self.host
            config['working_dir'] = self.working_dir
            config['username'] = self.username
            config['password'] = None if self.password is None else base64.b64encode(self.password.encode('utf-8'))
            config['pkey'] = None if self.pkey is None else base64.b64encode(self.pkey.encode('utf-8'))
            config['crimpl_env'] = self.crimpl_env
            config['env_name'] = self.env_name
            config['env_dir'] = self.env_dir
            config['extra_path'] = self.extra_path
            config['extra_ld_library_path'] = self.extra_ld_library_path

            json.dump(config, conffile)

        return fname

    def connect(self, expand_remote_home=True):
        if self.tunnel is not None:
            # tunnel already initialized, nothing to do.
            return

        self.tunnel = pm.SSHClient()
        if self.username is None and self.password is None and self.pkey is None:
            self.tunnel.load_system_host_keys()

        self.tunnel.connect(self.host, username=self.username, password=self.password, pkey=self.pkey)

        # now let's do the user a favor and expand the remote "~" if it occurs in self.working_dir:
        if expand_remote_home and '~' in self.working_dir:
            with self.tunnel.open_sftp() as uplink:
                remote_home = uplink.normalize('.')
            self.working_dir = self.working_dir.replace('~', remote_home)

    def disconnect(self):
        self.tunnel.close()
        self.tunnel = None

    def create_crimpl_env(self, crimpl_env=None, env_name=None, env_dir=None):
        crimpl_env = self.crimpl_env if crimpl_env is None else crimpl_env
        env_name = self.env_name if env_name is None else env_name
        env_dir = self.env_dir if env_dir is None else env_dir
        
        if crimpl_env != 'none' and env_name is None:
            raise ValueError(f'passing `env_name` is required when crimpl_env={crimpl_env}.')
        
        # test if the environment exists:
        if crimpl_env == 'none':
            # nothing to be done
            return
        elif crimpl_env == 'venv':
            try:
                # test is venv already exists:
                self.exec_command(f'ls {self.venv}/bin/activate')
            except RuntimeError:
                # it doesn't; create it:
                self.venv = os.path.join(env_dir, env_name)
                self.exec_command(f'python -m venv {self.venv}')
                self.crimpl_env = crimpl_env
                self.env_name = env_name
                self.env_dir = env_dir
                return
        elif crimpl_env == 'conda':
            raise NotImplementedError('conda not yet implemented.')
        else:
            raise ValueError(f'crimpl_env={crimpl_env} not recognized, aborting.')

    def exec_command(self, cmd, connect_if_needed=True, activate_crimpl_env=False):
        if self.tunnel is None and not connect_if_needed:
            raise ValueError('the ssh tunnel is not established. Please call <self>.connect() or pass `connect_if_needed=True`.')

        if self.tunnel is None and connect_if_needed:
            self.connect()

        prefix = ''

        if activate_crimpl_env:
            # we need to prepend the command with env activation:
            if self.venv:
                prefix = f'source {self.venv}/bin/activate; '
            elif self.conda:
                prefix = f'conda activate {self.conda}; '
            else:
                pass

        stdin, stdout, stderr = self.tunnel.exec_command(prefix + cmd)
        err = stderr.read().decode()
        if err:
            raise RuntimeError(err)
        
        cmd_output = stdout.read().decode().strip()
        return cmd_output

    def create_job(self, server, script, job_name=None, nnodes=1, nprocs=1, walltime='0-00:30:00', email=''):
        return SlurmJob(server=server, script=script, job_name=job_name, nnodes=nnodes, nprocs=nprocs, walltime=walltime, email=email)

    def submit_job(self, server, script, slurm_job_name=None, nnodes=1, nprocs=1, walltime='0-00:30:00', mail_user='', mail_type=None, addl_slurm_kwargs=None, files=None, crimpl_env='none', env_name=None, env_dir=None, isolate_env=False):
        self.connect()
        job = self.create_job(server=server, script=script, job_name=slurm_job_name, nnodes=nnodes, nprocs=nprocs, walltime=walltime, email=mail_user)
        job_dir = job.create_job_dir()
        for file in files:
            job.upload(file)
        slurm_script = job.create_slurm_script(fname='crimpl.sh', extra_path=self.extra_path, extra_ld_library_path=self.extra_ld_library_path)
        job.upload(os.path.abspath(slurm_script))
        self.exec_command(f'cd {job_dir}; sbatch crimpl.sh', activate_crimpl_env=True)
        return job


def list_servers(server_dir='~/.crimpl2/servers'):
    if os.path.exists(os.path.expanduser(server_dir)):
        return [conffile[:-5] for conffile in os.listdir(os.path.expanduser(server_dir))]
    return []


def load_server(name):
    filename = os.path.join(os.path.expanduser("~/.crimpl2/servers"), "{}.json".format(name))
    if not os.path.exists(filename):
        raise ValueError(f'could not find configuration at {filename}')
    with open(filename, 'r') as f:
        config = json.load(f)

    if 'crimpl' not in config.keys():
        raise ValueError('input configuration missing a `crimpl` entry')

    classname = config.pop('crimpl')
    config.pop('crimpl.version', None)

    return globals().get(classname)(**config)


class CrimplJob(object):
    def __init__(self):
        pass


class SlurmJob(CrimplJob):
    def __init__(self, server, script, job_name=None, nnodes=1, nprocs=1, walltime='0-00:30:00', email=''):
        if type(server) is not SlurmServer:
            raise ValueError(f'server needs to be a <SlurmServer> instance.')

        self.server = server
        self.script = script
        
        self.job_name = f'job_{dt.now().strftime("%Y%m%d-%H%M%S")}' if (job_name is None or job_name == '') else job_name
        self.job_dir = None
        
        self.nnodes = nnodes
        self.nprocs = nprocs
        self.walltime = walltime
        self.email = email

    def create_job_dir(self):
        if self.job_dir:
            return self.job_dir
        self.job_dir = os.path.join(self.server.working_dir, "jobs", self.job_name)
        self.server.exec_command(f'mkdir -p {self.job_dir}')
        return self.job_dir

    def create_slurm_script(self, fname, extra_path=None, extra_ld_library_path=None):
        script = f'#!/bin/bash\n\n' \
                 f'#SBATCH -D {self.job_dir}\n' \
                 f'#SBATCH -J {self.job_name}\n' \
                 f'#SBATCH -N {self.nnodes}\n' \
                 f'#SBATCH -n {self.nprocs}\n' \
                 f'#SBATCH -t {self.walltime}\n' \
                 f'#SBATCH --mail-type=START,END,FAIL\n' \
                 f'#SBATCH --mail-user={self.email}\n\n'

        if self.server.crimpl_env == 'venv':
            script += f'source {self.server.venv}/bin/activate\n\n'

        if self.server.crimpl_env == 'conda':
            script += f'conda activate {self.server.conda}\n\n'

        if extra_path:
            script += f'export PATH=$PATH:{extra_path}\n'

        if extra_ld_library_path:
            script += f'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{extra_ld_library_path}\n'

        script += f'mpirun python {self.script}'

        with open(fname, 'w') as f:
            f.write(script)

        return fname

    def upload(self, file):
        with self.server.tunnel.open_sftp() as uplink:
            print(f'uploading {file=} to {os.path.join(self.job_dir, os.path.basename(file))}')
            uplink.put(file, os.path.join(self.job_dir, os.path.basename(file)))

    def download(self, file):
        with self.server.tunnel.open_sftp() as downlink:
            print(f'downloading {os.path.join(self.job_dir, os.path.basename(file))} to {file}')
            downlink.get(os.path.join(self.job_dir, os.path.basename(file)), file)

    def job_status(self):
        output = self.server.exec_command(f'sacct -X --name {self.job_name} --format=jobid,elapsed,exitcode,qos,user,state')
        try:
            job_id, elapsed_time, exit_code, qos, user, state = output.split('\n')[-1].split()
        except ValueError:
            raise ValueError(f'the server does not know anything about {self.job_name=}.')

        return {'job_id': int(job_id), 'elapsed_time': elapsed_time, 'exit_code': exit_code, 'qos': qos, 'user': user, 'state': state}
