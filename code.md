<!-- #### To use the code.


Premeir: we understand the litho modelling.  -->


## Table of Contents

- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Step 0: Install Requirements](#step-0-install-requirements)
- [Step 1: Litho Dataset Generation](#step-1-litho-dataset-generation)
- [Step 2: Train the Neural litho simulator to overfit the litho process](#step-2-train-the-neural-litho-simulator-to-overfit-the-litho-process)
- [Step 3 (Optional): Use neural litho model in the downstream computational optics tasks](#step-3-optional-use-neural-litho-model-in-the-downstream-computational-optics-tasks)
- [License\&Citation](#licensecitation)

## Introduction

In what follows we shows how to use the code in the repo to reproduce the method in our neural litho paper. 

## Step 0: Install Requirements
To use the Neural Lens Modeling implementation, you need Python (version 3.7 or higher). Python environment including the required dependencies. This requires an NVIDIA GPU.

You can install the required Python packages using pip:

```
pip install -r requirements.txt
```

Or  install the packages when necessary (**Recommended**).

## Step 1: Litho Dataset Generation

To generate a dataset for digitalize a real-world litho process, follow these steps:

1. Prepare the random designed masks so as to explore the modelling of a litho system. We emprically find ~100 patterns would suffice when we tested on TPL.
2. Execectute the masks in the litho masks and get prints after the necessary procedures; Measure the prints in a high-definition 3D imaging system (in our case, the AFM).
3. Register the masks/prints pairs with homography. 
4. Save the registered masks/prints pairs to form you own dataset.



## Step 2: Train the Neural litho simulator to overfit the litho process 

To train the neural networks using the generated or captured dataset, follow these steps:

1. Load the dataset collected in the above step.
2. Split the dataset into training and validation sets. The `afm_dataio.py` helps this.  
3. Run the training in `main_fwd_litho_training.py`.
4. After Training, the checkpoint is saved, which is the best fits to represent litho system and can be used for downstream tasks.


## Step 3 (Optional): Use neural litho model in the downstream computational optics tasks 

To use the learned litho model in the downstream tasks, such as the computational optics tasks (holography optical elements and imaging lens) demonstrated in the paper, pls follow these steps:

1. Set the `use_litho_model_flag=True` in task params.  
2. Run either `main_inv_holo_optim.py` or `main_inv_lens_optim.py` for HOE or Lens design for direct or computational imaging.
3. After the optimization, we get the printable design where the design considers both manufacturability and task metrics.


## License&Citation

The Neural Litho implementation is licensed under the [MIT License](LICENSE). Feel free to use and modify the code for your own purposes. 

If you find our work or any of our materials useful, please cite our paper:
```
@article{zheng2023neural,
            title={Neural Lithography: Close the Design-to-Manufacturing Gap in Computational Optics with a'Real2Sim'Learned Photolithography Simulator},
            author={Zheng, Cheng and Zhao, Guangyuan and So, Peter TC},
            journal={arXiv preprint arXiv:2309.17343},
            year={2023}
            }
```


```
@inproceedings{zheng2023close,
            title={Close the Design-to-Manufacturing Gap in Computational Optics with a'Real2Sim'Learned Two-Photon Neural Lithography Simulator},
            author={Zheng, Cheng and Zhao, Guangyuan and So, Peter},
            booktitle={SIGGRAPH Asia 2023 Conference Papers},
            pages={1--9},
            year={2023}
            }
```