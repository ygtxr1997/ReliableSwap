# Heatmap Regression via Randomized Rounding
----

## Test on [300W](https://ibug.doc.ic.ac.uk/resources/300-W/) (68 2D facial landmarks)

````bash
python examples/test_300w.py --model models/300w/hrnet18_256x256_p1/
````

| Backbone | BBox | Resolution | #Params | FLOPs | NME (%)|
|:--:|:--:|:--:|:--:|:--:|:--:|
| HRNet-W18 | P1 | 256x256 | 9.67M | 4.77G | 3.48 |
| HRNet-W18 | P2 | 256x256 | 9.67M | 4.77G | 3.62 |
| MobileNetV2 | P2 | 256x256 | 0.59M | 0.47G | 3.96 | 
| MobileNetV2 | P2 | 128x128 | 0.59M | 0.12G | 4.17 |


## Test on [AFLW2000-3D](http://www.cbsr.ia.ac.cn/users/xiangyuzhu/projects/3DDFA/main.htm) (68 3D facial landmarks)

````bash
python examples/test_aflw2000.py --model models/300wlp/hrnet18_256x256_p2/
````

| TrainSet | Backbone | BBox | Resolution | #Params | FLOPs | NME (%)|
|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| 300W-LP | HRNet-W18 | P2 | 128x128 | 9.67M | 1.19G | 3.04 |
| 300W-LP | MobileNetV2 | P2 | 128x128 | 0.59M | 0.12G | 3.21 |


