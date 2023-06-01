ep_num=600
seed=50

# CloserLook Models
for method in "maml_approx" "relationnet" "protonet" "matchingnet"
do
  for model_name in 'Conv4' 'Conv6' 'ResNet10' 'ResNet18' 'ResNet34'
  do
    for data_set in "base" "val"
    do
        python inference.py --method $method --data_set $data_set --ep_num $ep_num --seed $seed --model_name $model_name
    done
  done
done

# Simpleshot Models
for model_name in 'Conv4' 'Conv6' 'ResNet10' 'ResNet18' 'ResNet34' 'DenseNet121' 'WideRes'
do
  for data_set in "base" "val"
  do
      python inference.py --method simpleshot --data_set $data_set --ep_num $ep_num --seed $seed --model_name $model_name
  done
done

# DeepEMD
for data_set in "base" "val"
do
  python inference.py --method DeepEMD --data_set $data_set --ep_num $ep_num --seed $seed
done
