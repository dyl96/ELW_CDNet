python ./tools/train.py --file_root LEVIR --lr 5e-4 --max_steps 40000 --batch_size 32 --gpu_id 0

# python ./tools/train.py --file_root BCDD --lr 5e-4 --max_steps 40000

python ./tools/train.py --file_root SYSU --lr 5e-4 --max_steps 40000 --batch_size 32 --gpu_id 0

#python ./tools/train.py --file_root DSIFN --inWidth 512 --inHeight 512 --lr 5e-4 --max_steps 80000 --batch_size 32 --gpu_id 1