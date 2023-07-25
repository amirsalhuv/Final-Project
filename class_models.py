import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from torch import nn
import cv2
from torchvision import models


class CombinedModel(nn.Module):
    def __init__(self, yolo_model, patch_model):
        super().__init__()
        self.yolo_model = yolo_model.eval()
        self.patch_model = patch_model.eval()
        self.test_image_transforms = transforms.Compose([
              transforms.Resize((224, 224)),
              transforms.ToTensor(),
              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
          ])

    def forward(self, input):
        predicted_mask  = np.zeros((input.shape[0],input.shape[1]),dtype=np.uint8)

        with torch.no_grad():  # Disables gradient calculation to save memory
            yolo_out = self.yolo_model(input)

            # Get all bouding boxes
            for *box, _, _ in yolo_out.pred[0]:
              # Get the bounding box (and normilize them)
              x, y, w, h = (int(box[0]),int(box[1]),int(box[2]),int(box[3]))

              # Cut the input according to the window
              patch = input[y:h,x:w,:]

              # Resize according to the input for the 2nd model
              resized_patch = cv2.resize(patch,(256,256))

              # Convert the numpy array to a PIL Image
              output = Image.fromarray(resized_patch.astype('uint8'))

              # Convert to patch input shape
              output = self.test_image_transforms(output)
              output = torch.unsqueeze(output,0)

              # Inference from patch model
              output = self.patch_model(output)


              # Revert back to the mask size
              output = torch.squeeze(output)
              output = (output.cpu().numpy()>0.5)
              output = output.astype(np.uint8)

              # return back to original size
              output = cv2.resize(output,(w-x,h-y))

              # Add the output to the total detected mask
              predicted_mask[y:h,x:w]= output


            '''
            fig, ax = plt.subplots(2, 3, figsize=(6, 6))
            ax = ax.flatten()
            ax[0].imshow(patch)
            ax[0].title.set_text('patch')
            ax[1].imshow(input,cmap = 'gray')
            ax[1].title.set_text('input')
            ax[2].imshow(resized_patch,cmap = 'gray')
            ax[2].title.set_text('resized_patch')
            ax[3].imshow(targets,cmap = 'gray')
            ax[3].title.set_text('GT')
            ax[4].imshow(predicted_mask,cmap = 'gray')
            ax[4].title.set_text('predicted_mask')
            plt.tight_layout()
            plt.show()
            '''

        return predicted_mask
    
def double_conv(in_channels, out_channels):
      return nn.Sequential(
          nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
          nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True)
      )


def up_conv(in_channels, out_channels):
      return nn.ConvTranspose2d(
          in_channels, out_channels, kernel_size=2, stride=2
      )


class VGGUnet(nn.Module):
    #  VGG-16 (with BN) encoder.

      def __init__(self, encoder, *, pretrained=False, out_channels=1):
          super().__init__()

          self.encoder = encoder(pretrained=pretrained).features
          self.block1 = nn.Sequential(*self.encoder[:6])
          self.block2 = nn.Sequential(*self.encoder[6:13])
          self.block3 = nn.Sequential(*self.encoder[13:20])
          self.block4 = nn.Sequential(*self.encoder[20:27])
          self.block5 = nn.Sequential(*self.encoder[27:34])

          self.bottleneck = nn.Sequential(*self.encoder[34:])
          self.conv_bottleneck = double_conv(512, 1024)

          self.up_conv6 = up_conv(1024, 512)
          self.conv6 = double_conv(512 + 512, 512)
          self.up_conv7 = up_conv(512, 256)
          self.conv7 = double_conv(256 + 512, 256)
          self.up_conv8 = up_conv(256, 128)
          self.conv8 = double_conv(128 + 256, 128)
          self.up_conv9 = up_conv(128, 64)
          self.conv9 = double_conv(64 + 128, 64)
          self.up_conv10 = up_conv(64, 32)
          self.conv10 = double_conv(32 + 64, 32)
          self.conv11 = nn.Conv2d(32, out_channels, kernel_size=1)

      def forward(self, x):
          block1 = self.block1(x)
          block2 = self.block2(block1)
          block3 = self.block3(block2)
          block4 = self.block4(block3)
          block5 = self.block5(block4)

          bottleneck = self.bottleneck(block5)
          x = self.conv_bottleneck(bottleneck)

          x = self.up_conv6(x)
          x = torch.cat([x, block5], dim=1)
          x = self.conv6(x)

          x = self.up_conv7(x)
          x = torch.cat([x, block4], dim=1)
          x = self.conv7(x)

          x = self.up_conv8(x)
          x = torch.cat([x, block3], dim=1)
          x = self.conv8(x)

          x = self.up_conv9(x)
          x = torch.cat([x, block2], dim=1)
          x = self.conv9(x)

          x = self.up_conv10(x)
          x = torch.cat([x, block1], dim=1)
          x = self.conv10(x)

          x = self.conv11(x)

          x = torch.sigmoid(x)

          return x

class Classifer_model(nn.Module):
      def __init__(self):
        super().__init__()

        # Load pre-trained VGG16 model and adjust the final layer
        self.classifer_model = models.vgg16(pretrained=True)

        #print(models.vgg16_bn(pretrained=True))

        # Change the final layer of VGG16 Model for Transfer Learning
        self.classifer_model.classifier[6] = torch.nn.Sequential(
            torch.nn.Linear(4096, 256),
            torch.nn.BatchNorm1d(256),  # Batch Normalization layer
            torch.nn.ReLU(),
            torch.nn.Dropout(0.4),
            torch.nn.Linear(256, 2), # Since we have two classes
            torch.nn.LogSoftmax(dim=1) # For using NLLLoss()
          )

      def forward(self, x):


        x = self.classifer_model(x)

        return x
      
