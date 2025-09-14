# Clear2Fog: A Multimodal End-to-End Pipeline for Fog Simulation

This repository contains the source code for the MSc dissertation, "Towards Robust 3D Object Detection in Fog: A Multimodal End-to-End Pipeline for Fog Simulation".

**Dissertation link:** [PDF](https://drive.google.com/file/d/1ync_1Tntt8n_aJwgNUptoDkEv8AkEp5P/view?usp=drive_link)

---
To address the critical scarcity of large-scale foggy datasets for autonomous vehicle research, this dissertation introduces Clear2Fog (C2F), a configurable and reusable pipeline for generating consistent multimodal fog. 
The pipeline introduces key innovations, including a physically-grounded atmospheric light model and a metric-driven Optimal Candidate Selection (OCS) module to enhance realism. 
A comprehensive evaluation revealed a critical trade-off between the OCS module's improved global realism and the introduction of local structural artifacts. 
Ultimately, the C2F pipeline provides a deeper understanding of the complexities of simulation realism and offers a flexible tool for both large-scale data generation (via its baseline) and in-depth validation (via the OCS module).

### Key Features

* **Multimodal:** Generates consistent fog on multimodal (camera + LiDAR) datasets, and can also process camera-only or LiDAR-only data
* **Configurable:** Allows users to set a specific fog density via a visibility parameter
* **Realistic:** Incorporates a novel OCS module to address unrealistic shadows and lighting in standard models
* **Generalisable:** Proven to work on datasets outside the autonomous driving domain like COCO and Flickr30k
* **Practical & Easy to Use:** Get started in three simple steps: create the environment from the .yml file, download the required pre-trained models and run the pipeline

---
### Getting Started

#### 1. Setup Environment

Clone the repository and create the Conda environment from the provided file. This will install all necessary dependencies.

```bash
git clone https://github.com/mmohamed28/Clear2Fog
cd Clear2Fog
conda env create -f environment.yml
conda activate c2f
```

#### 2. Download Pre-trained Models

This pipeline relies on several pre-trained models. Please download the following checkpoints and place them in the specified directories:

* Depth Pro: Download ```depth_pro.pt``` from [ml-depth-pro](https://github.com/apple/ml-depth-pro/blob/main/get_pretrained_models.sh) and place it in ./src/fog_pipeline/depth_map/depth_pro/checkpoints/
* DHAN (De-shadowing):
  * Download the pre-trained ```SRD+ models``` from [ghost-free-shadow-removal](https://github.com/vinthony/ghost-free-shadow-removal) and ```imagenet-vgg-verydeep-19``` from [MatConvNet](https://www.vlfeat.org/matconvnet/pretrained/#downloading-the-pre-trained-models) and place them in ./src/fog_pipeline/image_enhancement/DHAN/models
  * The exact models used when testing can be found [here](https://drive.google.com/file/d/1pcrrFMs0jEUc0wIGzNQlzY5MAWS6-br6/view?usp=sharing)

#### 3. Demo
You are now ready to run the pipeline. You can use the following command as a template:

```bash
python c2f.py \
-c CAMERA_DIR \
-l LIDAR_DIR \
-o OUTPUT_DIR \
-v VIS_METRES \
--no_ocs
```
* ```-c```: Input folder containing the RGB files
* ```-l```: Input folder containing the ```.npy``` point cloud files
* ```-o```: Empty output folder to save the simulated files
* ```-v```: Visibility distance in metres, controlling fog density (lower values = denser fog)
* ```--no_ocs```: A flag to disable the Optimal Candidate Selection module and run the baseline pipeline

---
### Validation: The Waymo Open Dataset
The pipeline's end-to-end functionality was validated on a scene from the Waymo Open Dataset using both camera and LiDAR data. The full output can be accessed here: COMING SOON

---
### Acknowledgments and Dependencies
This project builds upon the excellent work of many other researchers. The ```LICENSES``` folder contains the full license for each of the third-party models used.

* Depth Pro: [https://github.com/apple/ml-depth-pro](https://github.com/apple/ml-depth-pro)
* DHAN: [https://github.com/cydiachen/DHAN](https://github.com/vinthony/ghost-free-shadow-removal)
* SCI: [https://github.com/vis-opt-group/SCI](https://github.com/vis-opt-group/SCI)
* LiDAR Fog Simulation: [https://github.com/MartinHahner/LiDAR-Fog-Simulation](https://github.com/MartinHahner/LiDAR_fog_sim)
* AuthESI: https://github.com/noahzn/FoHIS
