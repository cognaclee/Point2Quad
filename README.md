# Point2Quad
**Point2Quad: Generating Quad Meshes from Point Clouds via Face Prediction**

## Introduction
![Full pipeline of our Point2Quad](assets/pipeline.png)
In this repository, we release the code for Point2Quad, a learning-based method for quad-only mesh generation from point clouds. The key idea is learning to identify quad mesh with fused pointwise and facewise features. Specifically, Point2Quad begins with a k-NN-based candidate generation considering the coplanarity and squareness. Then, two encoders are followed to extract geometric and topological features that address the challenge of quad-related constraints, especially by combining in-depth quadrilaterals-specific characteristics. Subsequently, the extracted features are fused to train the classifier with a designed compound loss. The final results are derived after the refinement by a quad-specific post-processing. Extensive experiments on both clear and noise data demonstrate the effectiveness and superiority of Point2Quad, compared to baseline methods under comprehensive metrics. 

### Update 29/03/2025: New [PyTorch implementation](https://github.com/cognaclee/Point2Quad) available. 


## Citation
If you find our work useful in your research, please consider citing:

```
@inproceedings{li2025nopain,
  title={NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary},
  author={Li, Zezeng and Du, Xiaoyu and Lei, Na and Chen, Liming and Wang, Weimin},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```


## Performances

We provide scripts for many experiments. The instructions to run these experiments are in the [doc](./doc) folder.

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


### Prepare Data and Models

1. **Download the datasets and place them in the `data/` directory:**

	```
	data/
	├── shapenetcore_partanno_segmentation_benchmark/
	├── ScanObjectNN/
	└── modelnet40_normal_resampled/
	```
2. **Download the [pretrained models](https://drive.google.com/drive/folders/1K0i1Q-77maDBT03fSGRQzHXA1bvgNSD5?usp=drive_link) and place them in the `pretrained/` directory:**
	```
	# Create the pretrained directory if it doesn't exist
	mkdir -p pretrained
	```

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
        ## Note that the quad mesh data needs to be in .obj format.
	python test.py
	```

## Acknowledgment

Our code uses <a href="https://github.com/HuguesTHOMAS/KPConv">KPConv</a> as the backbone.

## License
Our code is released under MIT License (see LICENSE file for details).

