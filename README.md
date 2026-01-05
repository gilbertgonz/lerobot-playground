# lerobot-playground
Adhoc playground for RL prototyping on the HuggingFace LeRobot stack

## how to record data via phone
1. open app in phone
2. run `python3 phone_record.py`
3. follow terminal instructions (space bar for recording an episode)
4. data stored in ./output

## how to view recorded episodes
https://huggingface.co/spaces/lerobot/visualize_dataset

Enter repo_id: `gilbertgonz/so101_training_data`

## how to train
```
python3 lerobot/src/lerobot/scripts/lerobot_train.py      --dataset.repo_id=gilbertgonz/so101_training_data      --policy.type=act      --output_dir=outputs/robot_learning_tutorial/act/      --steps=10000      --batch_size=16      --policy.device=cuda --policy.repo_id=gilbertgonz/so101_training_data```
```

## how to upload model
```
huggingface-cli upload gilbertgonz/so101-act-models  outputs/robot_learning_tutorial/act/checkpoints/010000/pretrained_model .
```

## how to teleop via handtracking
*NOT COMPLETE*

**need to implement IK (use phone teleop for this) and correctly map hand tracking actions to robot joints**

1. run `python3 handtracking_teleop.py`