#选择对应命令训练即可
nohup python -u train.py --model ResNet20    --amp  > logs/CIFAR10/train_ResNet20.log 2>&1 &
# nohup python -u train.py --model PlainNet20  --amp  > logs/CIFAR10/train_PlainNet20.log 2>&1 &
# nohup python -u train.py --model PlainNet32  --amp  > logs/CIFAR10/train_PlainNet32.log 2>&1 &
# nohup python -u train.py --model PlainNet44  --amp  > logs/CIFAR10/train_PlainNet44.log 2>&1 &
# nohup python -u train.py --model PlainNet56  --amp  > logs/CIFAR10/train_PlainNet56.log 2>&1 &
# nohup python -u train.py --model ResNet20    --amp  > logs/CIFAR10/train_ResNet20.log 2>&1 &
# nohup python -u train.py --model ResNet32    --amp  > logs/CIFAR10/train_ResNet32.log 2>&1 &
# nohup python -u train.py --model ResNet44    --amp  > logs/CIFAR10/train_ResNet44.log 2>&1 &
# nohup python -u train.py --model ResNet56  --deep_Network --weight_decay 0.0003 --amp  > logs/CIFAR10/train_ResNet56.log 2>&1 &
# nohup python -u train.py --model ResNet110 --deep_Network --weight_decay 0.0005 --amp  > logs/CIFAR10/train_ResNet110.log 2>&1 &