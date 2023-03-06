export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd src


# torchrun --nproc_per_node=8 main.py --model tf_efficientnetv2_s --lr 4e-4 --batch-size 8

# echo

torchrun --nproc_per_node=8 main.py

# echo

# # torchrun --nproc_per_node=8 main.py --model eca_nfnet_l0 --lr 2e-4 --batch-size 4

# # echo

# # torchrun --nproc_per_node=8 main.py --model tf_efficientnetv2_s --lr 4e-4