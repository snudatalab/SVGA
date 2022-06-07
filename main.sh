DATA="cora" # Choose a dataset used in the paper.
SEED=0      # Run the script for ten different seeds in [0, 9].
Y_RATIO=0   # Set this value in (0, 1] to use observed labels.

cd src

if [[ $DATA == "cora" ]]; then
  python main.py \
    --data $DATA \
    --y-ratio $Y_RATIO \
    --emb-norm unit \
    --dec-bias False \
    --x-loss balanced \
    --hidden-size 256 \
    --lamda 1 \
    --beta 0.1 \
    --lr 1e-3 \
    --dropout 0.5 \
    --seed $SEED

elif [[ $DATA == "steam" ]]; then
  python main.py \
    --data $DATA \
    --y-ratio $Y_RATIO \
    --emb-norm none \
    --dec-bias True \
    --x-loss balanced \
    --hidden-size 256 \
    --lamda 1 \
    --beta 0.1 \
    --lr 5e-3 \
    --dropout 0.0 \
    --seed $SEED

elif [[ $DATA == "pubmed" ]]; then
  python main.py \
    --data $DATA \
    --y-ratio $Y_RATIO \
    --emb-norm unit \
    --dec-bias False \
    --x-loss gaussian \
    --hidden-size 512 \
    --lamda 0.01 \
    --beta 1.0 \
    --lr 1e-3 \
    --dropout 0.5 \
    --seed $SEED

elif [[ $DATA == "coauthor" ]]; then
  python main.py \
    --data $DATA \
    --y-ratio $Y_RATIO \
    --emb-norm none \
    --dec-bias True \
    --x-loss gaussian \
    --hidden-size 512 \
    --lamda 0.01 \
    --beta 0.1 \
    --lr 1e-3 \
    --dropout 0.0 \
    --seed $SEED

elif [[ $DATA == "arxiv" ]]; then
  python main.py \
    --data $DATA \
    --y-ratio $Y_RATIO \
    --emb-norm unit \
    --dec-bias False \
    --x-loss gaussian \
    --hidden-size 512 \
    --lamda 0.1 \
    --beta 1.0 \
    --lr 1e-3 \
    --dropout 0.0 \
    --seed $SEED

else
  python main.py \
    --data $DATA \
    --y-ratio $Y_RATIO \
    --emb-norm unit \
    --dec-bias False \
    --x-loss balanced \
    --hidden-size 256 \
    --lamda 0.1 \
    --beta 0.1 \
    --lr 1e-3 \
    --dropout 0.5 \
    --seed $SEED
fi
