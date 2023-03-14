from smol import Smol

q = Smol.load_from_checkpoint('models\epoch=6-step=100.ckpt')

print(q)