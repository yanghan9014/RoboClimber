# RoboClimber

## Usage
First setup the environment:
```bash
conda create -n RL_final python=3.10 
conda activate RL_final
pip install -r requirements.txt
pip install -e .
```

You can test the custom environment by running
```
python climb/scripts/test_train.py
```
Which should generate a video in the `videos/` directory

Run a `humanoid-v5` environment with 
```
python3 climb/scripts/test_PPO.py --max_training_timesteps 1000000 --exp_name ep_len1000_lr3e-4_layer8_s64 -lr 3e-4 -l 8 -s 64 --ep_len 1000
```
(`HalfCheetah-v5` works quite well with the current implementation)

Run the custom `climber-v0` environment with 
```
python3 climb/scripts/test_PPO.py --max_training_timesteps 300000  --env_name Climber-v0 --xml_file 'assets/climber_v0.xml' --keyframe 'foot_up' --exp_name lr5e-4_layer8_s64 -lr 5e-4 -l 8 -s 64 --save_params
```
You might need to tune the hyperparameters, and the loss computation for critic (`value_loss`) is dubious at the moment


## Problems
- `mujoco-py` is deprecated and isn't supported on M1 mac; `stable-baselines3` requires `gymnasium`<0.30 or >=0.28.1, but `gymnasium` still requires `mujoco-py` until the latest `v1.0.0`, so its the only option for me now (Daniel)

## TODOs
- [x] Register custom environment `Climber`
- [x] Add climbing shoes to the `climber-v0.xml` model
- [ ] Add fingers to the `climber-v0.xml` model
- [ ] Modify action space & observation space of `Climber`
- [ ] Tweek reward & terminate condition of `Climber`
- [ ] Implement PPO
- [ ] Test PPO on `humanoid-v5`, training it to walk forward
- [ ] Generate climbing wall .xml file

## How to Use Pre-commit Hooks: A Step-by-Step Guide (TQ ChatGPT)
### 1. Ensure You Have Pre-commit Installed
First, ensure that `pre-commit` is installed. You can check if it's already installed by running:

```bash
pre-commit --version
```

If it’s not installed, you can install it using pip:
```bash
pip install pre-commit
```

### 2. How to Run Pre-commit Hooks
To run the pre-commit hooks manually on all files in the repository, you can use:
```bash
pre-commit run --all-files
```
This command will execute all the hooks configured in `.pre-commit-config.yaml` on all the files in the repository.

### 3. Handling Failed Checks

If any pre-commit hook fails, here’s what you should do:
- Fix the Issues: The pre-commit hooks will provide feedback on what needs to be fixed (e.g., formatting issues, linting errors, etc.). Follow the feedback to fix your code.
- Auto-formatting (if applicable): If a formatter (like `black`) reports an issue, it will automatically format the code for you. 

### 3. TL;DR
#### Run pre-commit hooks on all files:
```bash
pre-commit run --all-files
```
#### Commit changes (with pre-commit hooks):
```bash
git add "Some files"
git commit -m "Your commit message"
```