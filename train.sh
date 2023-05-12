for method in protonet matchingnet relationnet_softmax maml_approx
do
    for model in Conv4 ResNet10 ResNet18 ResNet34 ResNet50
    do
        python train.py --dataset miniImagenet --model $model --method $method --stop_epoch 600 --n_shot 1 --resume
    done
done

