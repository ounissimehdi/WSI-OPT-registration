<p align="center">
 <img width="400" src="./demo/point2Cell.png">
</p>
<h1 align="center">
 Point2Cell:
 
 low-shot learning for interactive biomedical image annotation. Instantiation to stain-free phase-contrast microscopy 
</h1>
<p align="center">
   <img width="256" height="256" src="./demo/microglial_cells_demo_1024.gif">
   <img width="256" height="256" src="./demo/hela_demo_1024.gif">
</p>

## 🛠️ Installation Steps

1. Install anaconda-python (for more details please check https://docs.anaconda.com/anaconda/install/index.html)

2. Clone the repository (this will take some time)

```bash
git clone https://github.com/ounissimehdi/Point2Cell
```

3. Change the working directory

```bash
cd Point2Cell
```

4. Create the point2Cell conda environment with all the dependencies

```bash
conda env create -f environment.yml
```
**Note** that this project is tested on: Windows: 11, MacOS: BigSur and Lunix: Ubuntu 20.04.3 LTS.
 
with the last version of Pytorch ( pytorch 1.10.2:cuda 11.3 and cudnn 8.0 ) to this date.

It supports GPU 😊

5. Activate the point2Cell conda environment

```bash
conda activate point2Cell
```

🌟 You are all set!


email address : daniel.racoceanu@icm-institute.org
```bash
@misc{Github,
  author={Mehdi Ounissi and Daniel Racoceanu},
  title={Point2Cell: low-shot learning for interactive biomedical image annotation. Instantiation to stain-free phase-contrast microscopy},
  year={2022},
  url={https://github.com/ounissimehdi/Point2Cell},
}
```

new_dataset