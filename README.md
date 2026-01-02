# lerobot-playground
Adhoc playground for RL prototyping on the HuggingFace LeRobot stack

## how to record data via phone
1. open app in phone
2. run `python3 phone_record.py`
3. follow terminal instructions (space bar for recording an episode)
4. data stored in ./output

## how to view recorded episodes
```
lerobot-dataset-viz --repo-id gilberto/so101_training_data --root outputs/datasets --episode-index 1
```

## how to train (locally)
1. reduce batch size to 8
2. use torch autocast
3. run: 
```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 lerobot/examples/tutorial/act/act_training_example.py
```

## how to teleop via handtracking
*NOT COMPLETE*

**need to implement IK (use phone teleop for this) and correctly map hand tracking actions to robot joints**

1. run `python3 handtracking_teleop.py`