ep_num=1000
seed=42
# for method in "simpleshot_conv4" "simpleshot_resnet10" "simpleshot_resnet18" "simpleshot_resnet34" "simpleshot_resnet50" "simpleshot_wideres" "simpleshot_densenet121"
for method in "relationnet_softmax"
do
  for model_name in 'Conv6'
  do
    for data_set in "base" "val" "novel"
    do
        python inference.py --method $method --data_set $data_set --ep_num $ep_num --seed $seed --model_name $model_name
    done
  done
done