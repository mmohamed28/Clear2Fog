# Clear2Fog

Clear2Fog (C2F) is a multimodal simulation pipeline built to address the lack of adverse weather data in autonomous driving. It transforms standard clear-weather data into consistent, physics-based fog across both camera and LiDAR simultaneously.

Instead of just adding a foggy filter, C2F uses a physically-grounded atmospheric model to ensure environmental consistency. This repository contains the framework used for our research on simulation realism and large-scale data scaling.


### Key Features

* **Multimodal:** Generates consistent fog on multimodal (camera + LiDAR) datasets, and can also process camera-only or LiDAR-only data
* **Physically Grounded:** Implements an atmospheric scattering model to ensure realistic light distribution and depth-aware simulation
* **Configurable:** Allows users to set a specific fog density via a visibility parameter
* **Generalisable:** Proven to work on datasets outside the autonomous driving domain like COCO and Flickr30k
* **Practical & Easy to Use:** Get started in three simple steps: create the environment from the .yml file, download the required pre-trained model and run the pipeline

---
### Getting Started

#### 1. Setup Environment

Clone this repository and create the Conda environment from the provided file. This will install all necessary dependencies.

```bash
git clone https://github.com/mmohamed28/Clear2Fog
cd Clear2Fog
conda env create -f environment.yml
conda activate c2f
```

#### 2. Download Pre-trained Models

This pipeline relies on the Depth Pro model. Please download the following checkpoint and place it in the specified directory:

* Depth Pro: Download ```depth_pro.pt``` from [ml-depth-pro](https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh) and place it in ```./src/fog_pipeline/depth_map/depth_pro/checkpoints/```

#### 3. Demo
You are now ready to run the pipeline. You can use the following command as a template:

```bash
python c2f.py \
-c CAMERA_DIR \
-l LIDAR_DIR \
-o OUTPUT_DIR \
-v VIS_METRES
```
* ```-c```: Input folder containing the RGB files
* ```-l```: Input folder containing the ```.npy``` point cloud files
* ```-o```: Empty output folder to save the simulated files
* ```-v```: Visibility distance in metres, controlling fog density (lower values = denser fog)

---
### Acknowledgments and Dependencies
This project builds upon the excellent work of many other researchers. The ```LICENSES``` folder contains the full license for each of the third-party models used.

* Depth Pro: [https://github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
* LiDAR Fog Simulation: [https://github.com/MartinHahner/LiDAR-Fog-Simulation](https://github.com/MartinHahner/LiDAR_fog_sim)
* WaymoCOCO: [https://github.com/shinya7y/WaymoCOCO](https://github.com/shinya7y/WaymoCOCO)
