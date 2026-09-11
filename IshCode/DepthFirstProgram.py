#IMPORTS
import glob, os, torch, matplotlib.pyplot as plt, numpy as np
from depth_anything_3.api import DepthAnything3
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
print("all imports done")

#Now we just have to combine the results and find the average depth of a single mask.
#To find the average depth, just multiply the depth results by the mask.

#FIND IMAGE PATH
device = torch.device("cpu")
example_path = "Depth-Anything-3/assets/examples/SOH"
images = sorted(glob.glob(os.path.join(example_path, "*.png")))

#LOAD DATA FOR Depth
DepthData = [images[0]]
print("images loaded -- type is:")
print(type(images))

#RUN DEPTH
model = DepthAnything3.from_pretrained("depth-anything/DA3-SMALL")
model2 = model.to(device = device)
DepthPrediction = model.inference(
    DepthData,
)
print("done with Depth stuff")

#RUN SAM2
#SAMData = Image.open(images[0])
#SAMData = np.array(SAMData.convert("RGB"))
SAMData = DepthPrediction.processed_images[0]
print("SAM2 images loaded -- dimensions are", SAMData.shape)

checkpoint = "./sam2/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device = "cpu"))
print("loaded Sam model")
input_point = np.array([[424, 200]])
input_label = np.array([1])
#print(images[0].dtype)
with torch.inference_mode(), torch.autocast(device_type = "cpu", dtype=torch.bfloat16):
    predictor.set_image(SAMData)
    print("loaded onto sam!")
    SAM_predictions = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
print("Done with SAM predictions!")
print("sam prediction shape is: ", SAM_predictions[0][0].shape)
print("sam prediction type is: ")
print(type(SAM_predictions[0]))
print("sam prediction each element is: ")
print(SAM_predictions[0].dtype)
print("image shape is: ", SAMData.shape)

#finding average depth
depthMask = DepthPrediction.depth[0] * SAM_predictions[0][0]
averageDepth = np.mean(depthMask)
print("average depth is: ", averageDepth)

if __name__ == '__main__':
    #Goal == try getting average depth of an object.
    
    #Print basic test data about model
    print("Depth Image shape", DepthPrediction.processed_images.shape)
    print("Depth prediction shape", DepthPrediction.depth.shape)
    print("depth dtype: ")
    print(type(DepthPrediction.depth))
    print("depth max: ")
    print(DepthPrediction.depth.max())
    
    #PLT DISPLAY -- later you can try displaying depth map if you want
    xScatter = np.array([424])
    yScatter = np.array([200])
    ax = plt.gca()
    ax.scatter(xScatter, yScatter, s=375, color = 'green', marker = "*", edgecolors='black')

    maskColor = np.array([0, 255, 0])
    imsize = SAM_predictions[0].shape[-2:]#Should be (height, width)
    print("imsize = ", imsize)
    mask = SAM_predictions[0][0]
    print("mask maximum is: ", SAM_predictions[0].max())
    mask = mask.astype(np.uint8)
    mask_imagePLT = mask.reshape(imsize[0], imsize[1], 1) * maskColor.reshape(1, 1, -1)
    print("PLT mask shape", mask_imagePLT.shape)
    #ax.imshow(mask_imagePLT)
    
    plt.imshow(SAMData, cmap = "viridis", interpolation = "nearest")
    ax.imshow(mask_imagePLT, alpha=0.5)
    plt.colorbar()
    plt.title("image 0")
    plt.show()
    print("done with the stuff")
