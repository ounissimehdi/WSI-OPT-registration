<h1 align="center">
 Efficient 3D reconstruction of Whole Slide Images in Melanoma
</h1>

<p align="center">
 <img width="600" src="figure.png">
</p>

<h2 align="center">
Optimization scheme for whole slide images registration
</h2>

Please cite the full paper:
https://spie.org/medical-imaging/presentation/Efficient-3D-reconstruction-of-Whole-Slide-Images-in-Melanoma/12471-67?SSO=1

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

```bash
@misc{Github,
  author={J. Arslan, M. Ounissi, H. Luo, M. Lacroix, P. Dupre, P. Kumar, A. Hodgkinson, S.Dandou, R. Larive, C. Pignodel, L. Le Cam, O. Radulescu, and D. Racoceanu},
  title={Efficient 3D reconstruction of Whole Slide Images in Melanoma},
  year={2023},
  url={https://github.com/ounissimehdi/WSI-OPT-registration},
}
```