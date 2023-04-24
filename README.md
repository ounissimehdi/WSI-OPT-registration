<h1 align="center">
 Efficient 3D reconstruction of Whole Slide Images in Melanoma
</h1>

<p align="center">
 <img width="600" src="figure.png">
</p>

<h2 align="center">
Optimization scheme for whole slide images registration
</h2>

The full paper can be found on the follwing link: https://hal.science/hal-03834014.

Thank you in advance for citing the paper and the GitHub repository.

## 🛠️ Installation Steps

1. Install anaconda-python (for more details please check https://docs.anaconda.com/anaconda/install/index.html)

2. Clone the repository

```bash
git clone https://github.com/ounissimehdi/WSI-OPT-registration
```

3. Change the working directory

```bash
cd WSI-OPT-registration
```

4. Create a conda environment with all the dependencies from the requirements file, then activate your configured Python environment:

```bash
conda activate YOUR_ENV
```
**Note** that this project is tested on: Windows: 11, MacOS: BigSur and Lunix: Ubuntu 20.04.3 LTS.

🌟 You are all set!


## 🕯 How to use it with HPC (SLRUM)

1. Change the path to your HE images data directory in: HE_clean.py, para_reg_v3.py, reg_large.py and gif_creation.py

2. Run
```bash
bash dependency.sh
```
**Note** this will run the HE_clean.py using stage0.sh SLURM configuration then para_reg_v3.py from stage1.sh etc. SLURM configuration can be modified to suit your HPC specifications also the job array (the number of HE WSI).

### More details:

For the registration, to change the path to the *.tif files:

HE_clean.py: lines 9-10
para_reg_c3.py: lines 172-174
reg_large.py: lines 84, 91, 119
gif_creation: lines 9

The default setting used is to put all HE in a folder:
```bash
/path_to_your_dir/HE/HE_reg_dataset/YOUR.tif
/path_to_your_dir/HE/HE_dataset/YOUR.tif
```
This will be created by HE_clean.py:
```bash
/path_to_your_dir/HE/clean_dataset/YOUR.tif
```
This will be created by para_reg_v3.py:
```bash
/path_to_your_dir/HE/PAR_images_scale_(scale"between 10 and 100)/YOUR.csv
```
This will be created by reg_large.py:
```bash
/path_to_your_dir/HE/estimated_para_images_10to100/YOUR.csv
/path_to_your_dir/HE/estimated_para_images_10to100/YOUR.tif
/path_to_your_dir/HE/PAR_images_scale_'+str(scale)/YOUR.tif
```
To run all in one line in the cluster (keeping in mind that the code is on /path_to_your_dir/WSI-OPT-registration/YOUR.py) use:
```bash
bash dependency.sh
```
## 📽 Demos

### Before
<p align="center">
   <img width="400" src="before_registration_animation.gif">
</p>

### After (No ratio correction)
<p align="center">
   <img width="400" src="registration_animation_no_ratio.gif">
</p>

### After (with ratio correction)
<p align="center">
   <img width="400" src="registration_animation_ratio_corrected.gif">
</p>

## 🎁 Citation

```bash
@misc{Github,
  author={J. Arslan, M. Ounissi, H. Luo, M. Lacroix, P. Dupre, P. Kumar, A. Hodgkinson, S.Dandou, R. Larive, C. Pignodel, L. Le Cam, O. Radulescu, and D. Racoceanu},
  title={Efficient 3D reconstruction of Whole Slide Images in Melanoma},
  year={2023},
  url={https://github.com/ounissimehdi/WSI-OPT-registration},
}
```
```bash
@inproceedings{arslan:hal-03834014,
  TITLE = {{Efficient 3D reconstruction of Whole Slide Images in Melanoma}},
  AUTHOR = {Arslan, Janan and Ounissi, Mehdi and Luo, Haocheng and Lacroix, Matthieu and Dupr{\'e}, Pierrick and Kumar, Pawan and Hodgkinson, Arran and Dandou, Sarah and Larive, Romain M and Pignodel, Christine and Le~cam, Laurent and Racoceanu, Daniel and Radulescu, Ovidiu},
  URL = {https://hal.science/hal-03834014},
  BOOKTITLE = {{SPIE Medical Imaging 2023}},
  ADDRESS = {San Diego, California, United States},
  YEAR = {2023},
  MONTH = Feb,
  KEYWORDS = {cutaneous melanoma ; whole slide images ; hematoxylin and eosin ; 3D reconstruction ; vascular reconstruction ; personalized medicine},
  PDF = {https://hal.science/hal-03834014/file/abstract_submission.pdf},
  HAL_ID = {hal-03834014},
  HAL_VERSION = {v1},
}
```