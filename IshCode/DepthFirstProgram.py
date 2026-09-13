#IMPORTS
import glob, os, torch, matplotlib.pyplot as plt, numpy as np
from depth_anything_3.api import DepthAnything3
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image

#get device type
if torch.cuda.is_available():
    device_str = "cuda"
else:
    device_Str = "cpu"
#device_Str = xxx
device = torch.device(device_Str)
print(f"using device: {device}")


#FIND IMAGE PATH
example_path = "Depth-Anything-3/assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))

#LOAD DATA FOR Depth
DepthData = [images[0]]

#RUN DEPTH
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
model2 = model.to(device = device)
DepthPrediction = model.inference(
    DepthData,
)

#RUN SAM2
SAMData = DepthPrediction.processed_images[0]

checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device = device_Str))

input_point = np.array([[424, 200]])
input_label = np.array([1])
#print(images[0].dtype)
with torch.inference_mode(), torch.autocast(device_type = device_Str, dtype=torch.bfloat16):
    predictor.set_image(SAMData)
    SAM_predictions = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

#finding average depth
depthMask = DepthPrediction.depth[0] * SAM_predictions[0][0]
averageDepth = np.mean(depthMask)
print("average depth value of image: ", averageDepth)

if __name__ == '__main__':
    
    #PLT DISPLAY
    xScatter = np.array([424])
    yScatter = np.array([200])
    ax = plt.gca()
    ax.scatter(xScatter, yScatter, s=375, color = 'green', marker = "*", edgecolors='black')

    maskColor = np.array([0, 255, 0])
    imsize = SAM_predictions[0].shape[-2:]
    mask = SAM_predictions[0][0]
    mask = mask.astype(np.uint8)
    mask_imagePLT = mask.reshape(imsize[0], imsize[1], 1) * maskColor.reshape(1, 1, -1)
    #ax.imshow(mask_imagePLT)
    
    plt.imshow(SAMData, cmap = "viridis", interpolation = "nearest")
    ax.imshow(mask_imagePLT, alpha=0.5)
    plt.colorbar()
    plt.title("image 0")
    plt.show()
