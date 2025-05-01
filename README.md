# Point2Quad
**Point2Quad: Generating Quad Meshes from Point Clouds via Face Prediction**

## Introduction
![Full pipeline of our Point2Quad](assets/pipeline.png)
In this repository, we release the code for Point2Quad, a learning-based method for quad-only mesh generation from point clouds. The key idea is learning to identify quad mesh with fused pointwise and facewise features. Specifically, Point2Quad begins with a k-NN-based candidate generation considering the coplanarity and squareness. Then, two encoders are followed to extract geometric and topological features that address the challenge of quad-related constraints, especially by combining in-depth quadrilaterals-specific characteristics. Subsequently, the extracted features are fused to train the classifier with a designed compound loss. The final results are derived after the refinement by a quad-specific post-processing. Extensive experiments on both clear and noise data demonstrate the effectiveness and superiority of Point2Quad, compared to baseline methods under comprehensive metrics. 


## Citation
If you find our work useful in your research, please consider citing:

```
@ARTICLE{10945920,
  author={Li, Zezeng and Qi, Zhihui and Wang, Weimin and Wang, Ziliang and Duan, Junyi and Lei, Na},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={Point2Quad: Generating Quad Meshes from Point Clouds via Face Prediction}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Point cloud compression;Faces;Mesh generation;Feature extraction;Surface reconstruction;Surface fitting;Noise;Image reconstruction;Noise measurement;Accuracy;Mesh Generation;Quadrilateral;Point Clouds;Deep Learning},
  doi={10.1109/TCSVT.2025.3556130}}
```


## Performances

![qualitative.png](assets/qualitative.png)


## Usage

### Setup Environment


1. **Clone the repository and navigate to the project directory:**

   ```bash
   git clone https://github.com/cognaclee/Point2Quad.git
   cd Point2Quad
   ```
2. **Installation**

   A step-by-step installation guide for Ubuntu is provided in [INSTALL.md](./INSTALL.md). Windows is currently 
not supported as the code uses tensorflow custom operations.


### Data Process

1. **Download the [datasets](https://drive.google.com/drive/folders/1K0i1Q-77maDBT03fSGRQzHXA1bvgNSD5?usp=drive_link) and place them in the `data/` directory:**

	```
	data/
	├── shapenetcore_partanno_segmentation_benchmark/
	├── ScanObjectNN/
	└── modelnet40_normal_resampled/
	```
2. **Follow the [README.md](./datasets/DataProcess/README.md) in [DataProcess](./datasets/DataProcess) to process the data**

3. Use the script [m2h5.py](./datasets/m2h5.py) to convert the original .m file into an .h5 file for training.
### Run Point2Quad
1. **Train**
   
   Set ```point2QuadrilateralConfig.data_dir``` in ```train.py``` as **your data path**, then
   
	```bash
	python train.py
	```
3. **Test**
   
   Set ```config.data_dir``` in ```test.py``` as **your data path**, ```chosen_log``` as the **pretained model path**, then
   
	```bash
	python test.py
	```
 3. **Metrics Evaluation**
   
    Set ```input_dir``` in [cal_metrics.py](./utils/cal_metrics.py) as **the generated quad mesh path**, then
   
	```bash
	## Note that the quad mesh data needs to be in .obj format
	python test.py
	```

## Acknowledgment

Our code uses <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a> as the backbone.

Our dataset includes contributions from <a href="https://www.quadmesh.cloud/300/">quadmesh.cloud</a>, which provides high-quality quad meshes for our experiments.

## License
Our code is released under MIT License (see LICENSE file for details).

