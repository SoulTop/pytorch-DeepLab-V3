CUDA_VISIBLE_DEVICES=0 python test.py --backbone resnet --pattern test --workers 4 --test-batch-size 1 --gpu-ids 0 --checkname deeplab-resnet --eval-interval 1 --dataset pascal
