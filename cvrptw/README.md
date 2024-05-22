## GFACS with NLS for CVRPTW

### Dataset Generation
```raw
$ python utils.py
```
This will take up about 2GB of space.


### Training

The checkpoints will be saved in [`../pretrained/cvrptw`](../pretrained/cvrptw).

Train GFACS model for CVRPTW with `$N` nodes
```raw
$ python train.py $N
```


### Testing

Test GFACS for CVRPTW with `$N` nodes
```raw
$ python test.py $N -p "path_to_checkpoint"
```
