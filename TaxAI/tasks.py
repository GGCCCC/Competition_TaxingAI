import subprocess

from invoke import task

exec_path = "run.py"

@task()
def run(ctx):
    cmd = ["python", exec_path]
    print(cmd)
    with open("./log.log", "w") as f:
        subprocess.Popen(cmd, stdout=f, stderr=f, shell=False)

    
@task()
def stop(ctx):
    cmd = f'ps -ef | grep {exec_path}' + ' | grep -v grep | awk \'{print $2}\' | xargs kill -15'
    subprocess.Popen(cmd, shell=True)

    
@task()
def kill(ctx):
    cmd = f'ps -ef | grep {"python"}' + '  | grep hidden' + ' | grep -v grep | awk \'{print $2}\' | xargs kill -9'
    subprocess.Popen(cmd, shell=True)
    
@task()
def log(ctx):
    cmd = f'tail -f log.log'
    subprocess.run(cmd, shell=True)
    
@task()
def gpu(ctx):
    cmd = f'nvidia-smi'
    subprocess.run(cmd, shell=True)

