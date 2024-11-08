# RoboClimber

## Usage
First setup the environment:
```bash
conda create -n RL_final python=3.10 
conda activate RL_final
pip install -r requirements.txt
pip install -e .
```

Then you can run the following command to test training the model:
```
python climb/scripts/test_train.py
```

That should a video locates at `./videos/` folder.

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