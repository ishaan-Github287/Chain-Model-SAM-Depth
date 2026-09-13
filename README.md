# ChainModelSamDepth



I combined Depth Anything v3 and SAM 2 to create a program that determines the estimated average depth value of an object given a point in the image, and also displays the selected image mask in green on a separate screen.

## Acknowledgements:
  - Depth anything v3 for depth estimation
  - SAM 2 for object segmentation

## Independent usage

### Installing Chain-Model-Sam-Depth

```bash
git clone https://github.com/ishaan-Github287/Chain-Model-SAM-Depth.git
cd Chain-Model-Sam-Depth
```

### Other requirements
In order to run this independently, you also need to install Depth Anything v3 and SAM 2. The links to the exact repos, as well as some installation details are below. It's recommended you read these details in this section before proceeding to the installation instructions for Depth Anything v3 and SAM 2.

[Depth Anything v3](https://github.com/ByteDance-Seed/depth-anything-3)
  - Follow the basic installation instructions in the README of this repository. The optional gsplat/Gaussian Head installation is not required to run this project.

[SAM 2](https://github.com/facebookresearch/sam2)
  - Follow the basic installation instructions under the "installation" section of the README. Also Download the Checkpoints under the "Getting Started" section of the README

**Note**: This program assumes that the "sam2" and "Depth-Anything-3" folders (in which their respective repositories are cloned) are in the same directory as the "IshCode" folder (so within the "Chain-Model-SAM-Depth" folder), and that you are running the program from the Chain-Model-Sam-Depth directory.

```bash
Chain-Model-SAM-Depth/
├── IshCode/
│   └── DepthFirstProgram.py
├── sam2/
│   └── ...
└── Depth-Anything-3/
    └── ...
```

### Things to customize

  - This model uses the first ".png" image found in "Depth-Anything-3/assets/examples/SOH". If you would like to run the model with a different image, simply change the image path defined in the "DepthData" variable of "IshCode/DepthFirstProgram.py" to the desired image path.
  - The point on the image which selects the mask is defined as (424, 200). This can be changed on line 39 of IshCode/DepthFistProgram.py by replacing "424" with the desired x value in pixels and "200" with the desired y value in pixels. The point chosen is also visualized as a green star on top of the chosen image at the end of the program, in order to verify the point selection.
  - This program assumes that you are running that you are running the Machine Learning Models on a CPU or a CUDA-compatible GPU. If you would like to run the model on a different device, simply uncomment line 13 of "IshCode/DepthFirstProgram.py", where "device_Str" is redefined, and replace the "xxx" with a string containing the desired device type.

**Note**: If you use or build upon this project, please also cite the original works:
  - Lin et al., Depth Anything 3: Recovering the visual space from any views, 2025.
  - Ravi et al., SAM 2: Segment Anything in Images and Videos, 2024.
