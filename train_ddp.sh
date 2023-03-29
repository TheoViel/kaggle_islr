export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd src

torchrun --nproc_per_node=8 main.py 

# echo

# torchrun --nproc_per_node=8 main.py --lr 4e-4 --epochs 60

# echo

# torchrun --nproc_per_node=8 main.py --lr 4e-4 --epochs 80

