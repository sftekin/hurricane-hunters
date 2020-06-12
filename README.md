# hurricane-hunters
Forecasting of Tropical Cyclone Trajectories

## Installing

put the required data under data folder

Create Docker file
```
$ docker build -t hurr:v1 docker/.
```

build container
```
$ docker run -it --rm -v `pwd`:/workspace hur:v1
```

## Run

In order to train
```
$ python main.py --mode train 
```

to test
```
$ python main.py --mode test 
```
