set -x
# source activate style-transfer
python src/evaluate.py --checkpoint src/styles/scream.ckpt --in-path in.jpg --out-path out/scream-out.jpg
python src/evaluate.py --checkpoint src/styles/wreck.ckpt --in-path in.jpg --out-path out/wreck-out.jpg
python src/evaluate.py --checkpoint src/styles/rain-princess.ckpt --in-path in.jpg --out-path out/rain-princess-out.jpg
python src/evaluate.py --checkpoint src/styles/udnie.ckpt --in-path in.jpg --out-path out/udnie-out.jpg
python src/evaluate.py --checkpoint src/styles/la_muse.ckpt --in-path in.jpg --out-path out/la_muse-out.jpg
python src/evaluate.py --checkpoint src/styles/wave.ckpt --in-path in.jpg --out-path out/wave-out.jpg
# mv in.jpg out/in.jpg
# rm in.jpg
