# Setup

## Docker release build

These steps cover the **release** Docker image only.

### Prerequisites To Build Docker Image

- Docker + NVIDIA Container Toolkit installed and working.
- Isaac Sim downloaded and unzipped locally. [Follow the instructions to download the binary files](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html). Unzip it into desired folder, e.g. `/home/isaacsim/...`
  - Set `ISAACSIM_PATH` to the **unzipped** directory.
  - Example:
    - `export ISAACSIM_PATH=/path/to/isaacsim`
- Source the repo helpers in your shell startup so the functions are available:
  - `source /path/to/OmniDrones/helpers`
  - (Recommended) add this line to your `~/.bashrc`
- Ensure `OMNI_DRONES_DIR` points at your repo root (required by helpers):
  - `export OMNI_DRONES_DIR=/path/to/OmniDrones`
- Release builds clone a private repo during build, so you need an SSH agent. If you haven't already:
  - `ssh-add -l` should list keys
  - If not, run `ssh-add` first


From anywhere after sourcing `helpers`:

```
docker_build_omni latest
```

This stages a clean Isaac Sim copy and builds `omnidrones:latest`.

### Pull from dockerhub
Alternatively, you can pull the image directly from docker hub 
```
docker pull mjsucb/omnidrones:latest
```


### Run the release image

Run the release-specific helper:

```
docker_run_omni_release
```

### Start a default training run

Inside the container:

```
cd /workspace/omni_drones/scripts
python train.py algo=ppo headless=true wandb.mode=disabled
```

This launches the default PPO training in headless mode.