set -x
source activate style-transfer
python evaluate.py --checkpoint scream.ckpt --in-path in.jpg --out-path out/scream-out.jpg
python evaluate.py --checkpoint wreck.ckpt --in-path in.jpg --out-path out/wreck-out.jpg
python evaluate.py --checkpoint rain-princess.ckpt --in-path in.jpg --out-path out/rain-princess-out.jpg
python evaluate.py --checkpoint udnie.ckpt --in-path in.jpg --out-path out/udnie-out.jpg
python evaluate.py --checkpoint la_muse.ckpt --in-path in.jpg --out-path out/la_muse-out.jpg
mv in.jpg out/in.jpg
