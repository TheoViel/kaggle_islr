export OMP_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

cd src

torchrun --nproc_per_node=8 main_2.py

echo

<<<<<<< HEAD
# torchrun --nproc_per_node=8 main_2.py --mt-ema-decay 0.98

# echo
=======
# torchrun --nproc_per_node=8 main_2.py --epochs 120

# echo

>>>>>>> 3e30efd (mt + kd done)
