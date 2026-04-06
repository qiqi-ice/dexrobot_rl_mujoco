  nohup python scripts/train_grasp_ppo.py \
      --num-envs 24 \
      --rollout-steps 512 \
      --minibatch-size 2048 \
      --train-epochs 30 \
      --d-model 256 \
      --num-heads 8 \
      --num-layers 6 \
      --num-tokens 16 \
      --ffn-dim-factor 4 \
      --total-steps 10000000 \
      --device auto \
      > outputs/train_log.txt 2>&1 &
  echo "PID: $!"