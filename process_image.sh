set -x
source activate style-transfer
python evaluate.py --checkpoint scream.ckpt --in-path in.jpg --out-path scream-out.jpg
python evaluate.py --checkpoint wreck.ckpt --in-path in.jpg --out-path wreck-out.jpg
python evaluate.py --checkpoint rain-princess.ckpt --in-path in.jpg --out-path rain-princess-out.jpg
python evaluate.py --checkpoint udnie.ckpt --in-path in.jpg --out-path udnie-out.jpg
python evaluate.py --checkpoint la_muse.ckpt --in-path in.jpg --out-path la_muse-out.jpg
