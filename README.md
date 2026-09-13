# ChainModelSamDepth



I combined Depth Anything v3 and SAM 2 to create a program that determines the estimated average depth value of an object given a point in the image.

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
In order to run this independently, you also install Depth Anything v3 and SAM 2. This program assumes that the "sam2" and "Depth-Anything-3" folders (under which their respective repositories are cloned) are in the same directory as "IshCode" (so within the "Chain-Model-SAM-Depth" folder), and that you are running the program from the Chain-Model-Sam-Depth directory. The links to the exact repos, as well as some installation details are below:

[Depth Anything v3](https://github.com/ByteDance-Seed/depth-anything-3)
  - Follow the basic installation instructions in the README of this repository. The optional gsplat/Gaussian Head installation is not required to run this project.

[SAM 2](https://github.com/facebookresearch/sam2)
  - Follow the basic installation instructions under the "installation" section of the README. Also Download the Checkpoints under the "Getting Started" Section of the README

### Things to customize

  - This model uses the first image in "Depth-Anything-3/assets/examples/SOH". If you would like to run the model on a different image, simply change the image path defined in "DepthData" on line 23 of "IshCode/DepthFirstProgram.py" to the desired image path.
  - This program assumes that you are running that you are running the Machine Learning Models on a CPU or a CUDA-compatible GPU. If you would like to run the model on a different device, simply uncomment line 13 of "IshCode/DepthFirstProgram.py" and replace the "xxx" with a string containing the desired device type.

**Note**: If you use or build upon this project, please also cite the original works:
  - Lin et al., Depth Anything 3: Recovering the visual space from any views, 2025.
  - Ravi et al., SAM 2: Segment Anything in Images and Videos, 2024.
