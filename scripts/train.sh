#推荐使用amp和DALI加速训练
nohup python -u train.py --model ResNet18 --amp --DALI > logs/train_ResNet18.log 2>&1 &
# nohup python -u train.py --model PlainNet18  --amp --DALI > logs/train_PlainNet18.log 2>&1 &
# nohup python -u train.py --model PlainNet34  --amp --DALI > logs/train_PlainNet34.log 2>&1 &
# nohup python -u train.py --model ResNet34 --option A --amp --DALI > logs/train_ResNet34_A.log 2>&1 &
# nohup python -u train.py --model ResNet34 --option B --amp --DALI > logs/train_ResNet34_B.log 2>&1 &
# nohup python -u train.py --model ResNet34 --option C --amp --DALI > logs/train_ResNet34_C.log 2>&1 &
