import subprocess

def run_tensorboard(config):
    subprocess.Popen(["tensorboard", f"--logdir={config.log_dir}",
                  "--host=localhost","--port=6015"])

    print(f"run tensorboard:log_dir={config.log_dir} --host=localhost")