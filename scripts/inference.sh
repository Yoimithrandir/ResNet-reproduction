#10-crop评估top1 error与top5 error，自动加载best_model。推荐使用小batch size，避免10-crop内存溢出
python Inference.py --model ResNet18 --option A --batch_size 128
# python Inference.py --model PlainNet18  --batch_size 128
# python Inference.py --model PlainNet34  --batch_size 128
# python Inference.py --model ResNet18 --option B --batch_size 128
# python Inference.py --model ResNet18 --option C --batch_size 128
