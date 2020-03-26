import sys
import argparse
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch
import os
import numpy as np
import random
import math
import matplotlib
import matplotlib.pyplot as plt
import mrcnn.model as modellib
import coco
import tensorflow as tf

from mrcnn import utils
from mrcnn import visualize
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow,QMessageBox,QGraphicsPixmapItem,QGraphicsScene
from PyQt5.QtGui import QImage,QPixmap
from Demo import Ui_MainWindow
from skimage import io,transform,img_as_ubyte
from PIL import Image,ImageDraw,ImageFont
from torch.autograd import Variable
from network.Transformer import Transformer

global GANMODELPATH,ODMODELPATH,GANIMGPATH,ODIMGPATH,TESTIMGPATH,LOGPATH,MERIMGPATH,class_names
GANMODELPATH='./gantrained_model'
ODMODELPATH='./mrcnntrained_model'
GANIMGPATH='ganimg'
ODIMGPATH='odimg'
TESTIMGPATH='testimages'
LOGPATH='logs'
MERIMGPATH='merimg'
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyWindow, self).__init__(parent)
        self.setupUi(self)

    def setBrowerPath(self):
        filename,_ = QtWidgets.QFileDialog.getOpenFileName(self,"浏览","./","*.jpg")
        self.lineEdit.setText(filename)

    def readrawimage(self):
        filename = self.lineEdit.text()
        if filename:
            imgfile = io.imread(filename)
            gwidth = self.rawimgshow.geometry().width()           #width of the graph , same below
            gheight = self.rawimgshow.geometry().height()
            x = imgfile.shape[1]
            y = imgfile.shape[0]
            c = imgfile.shape[2]
            print("input_image size:",y,"*",x,"\n","channel:",c)
            self.zoomscale = min(gwidth/x,gheight/y)
            frame = QImage(imgfile, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.item.setScale(self.zoomscale)
            self.scene.addItem(self.item)
            self.rawimgshow.setScene(self.scene)


        else:
            QMessageBox.information(self,'error','no file to read,please choose a file')

    def tranimage(self):
        filename = self.lineEdit.text()            #image file path

        if filename:
        #---------------------------------------OD PART-----------------------------------------------
        # --------------------------------------------------------------------------------------------------
            config = InferenceConfig()

            # ------------------------------------------------------------------------------------------------
            # Local path to trained weights file
            COCO_MODEL_PATH = os.path.join(ODMODELPATH, "mask_rcnn_coco.h5")

            # Create model object in inference mode.
            model = modellib.MaskRCNN(mode="inference", model_dir=LOGPATH, config=config)

            # Load weights trained on MS-COCO
            model.load_weights(COCO_MODEL_PATH, by_name=True)
            print('od model loaded')
            # Load image
            rimage = io.imread(filename)
            odimage =rimage

            print('od image input done')
            height, width = odimage.shape[:2]

            # Run detection
            results = model.detect([odimage], verbose=0)
            print('image detected')
            # Visualize results====================================================================================================================
            r = results[0]

            #----------------------------------read bounding box & masks---------------------------------------------------------------------------
            boxes = r['rois']
            masks = r['masks']
            class_ids = r['class_ids']
            scores = r['scores']
            print('masks/boundingboxes loaded')
            # Number of instances
            N = boxes.shape[0]
            mer_image = np.ndarray([height,width,3]).astype('uint8')


            if not N:
                print("\n*** No instances to display *** \n")
            else:
                assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

            colors = visualize.random_colors(N)
            # Merge Image-----------------------------------------------------------
            for j in range(N):
                if not class_ids[j]==1:
                    continue
                for c in range(3):
                    mer_image[:, :, c] += masks[:,:,j] * rimage[:, :, c]

            antimer_image = rimage-mer_image
            #print(antimer_image.dtype)
            #------------------------------------------------------------------
            for i in range(N):
                color = colors[i]
                class_id = class_ids[i]
                score = scores[i] if scores is not None else None
                label = class_names[class_id]
                mask = masks[:, :, i]

                if not class_id==1:                  #category to draw
                    continue

                if not np.any(boxes[i]):
                    # Skip this instance. Has no bbox. Likely lost in image cropping.
                    continue

                # Bounding box-----------------------------------------
                y1, x1, y2, x2 = boxes[i]
                odimage = visualize.draw_box(odimage,boxes[i],255*np.array(color).astype('uint8'))
                print(str(i+1),' bounding box(es) drawed')
                #----------------------------------------------------------
                # Label


                caption = "{} {:.3f}".format(label, score-0.07) if score else label
                i_odimage = Image.fromarray(odimage)
                draw = ImageDraw.Draw(i_odimage)
                font = ImageFont.truetype("consola.ttf", 14, encoding="unic")
                draw.text((x1, y1+8), caption, fill=(255,255,255),font=font)
                odimage = np.array(i_odimage)
                print(str(i+1), ' caption(s) texted')

                #-------------------------------------------------------------------
                # Mask

                odimage = visualize.apply_mask(odimage, mask, color,alpha=0.3)


            #output image-----------------------------------------------------
            merfilename = os.path.join(MERIMGPATH, os.path.basename(filename)[:-4] + '_mer.jpg')
            odfilename = os.path.join(ODIMGPATH, os.path.basename(filename)[:-4] + '_od.jpg')
            anti_merfilename = os.path.join(MERIMGPATH, os.path.basename(filename)[:-4] + '_antimer.jpg')

            io.imsave(odfilename,odimage)
            io.imsave(merfilename, mer_image)
            io.imsave(anti_merfilename,antimer_image)

            #display image
            odimgfile = io.imread(odfilename)
            gwidth = self.odimgshow.geometry().width()  # width of the graph , same below
            gheight = self.odimgshow.geometry().height()
            x = odimgfile.shape[1]
            y = odimgfile.shape[0]

            self.zoomscale = min(gwidth / x, gheight / y)
            frame = QImage(odimgfile, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)

            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.item.setScale(self.zoomscale)
            self.scene.addItem(self.item)
            self.odimgshow.setScene(self.scene)

        #-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
            #======================================GAN PART=====================================================
            cartstyle = self.stylechoose.currentText()
            if cartstyle=='宫崎骏':
                artstyle='Hayao'
            elif cartstyle=='细田守':
                artstyle='Hosoda'
            elif cartstyle=='今敏':
                artstyle='Paprika'
            else:
                artstyle='Shinkai'

            # load model ------------------------------------------------------------------
            model = Transformer()
            model.load_state_dict(torch.load(os.path.join(GANMODELPATH,artstyle+ '_net_G_float.pth')))
            model.eval()


            print(artstyle,'model loaded ')

            #running mode
            model.cpu()
            #model.cuda()

            # load image---------------------------------------------------------------------

            input_image = Image.open(merfilename).convert("RGB")

            print('GAN_image input done')

            # resize image, keep aspect ratio

            h = input_image.size[0]
            w = input_image.size[1]

            load_size = max(h-2,w-2)            #change image size ,which is the same as input image now
            ratio = h * 1.0 / w
            
            if ratio > 1:
                h = load_size
                w = int(h * 1.0 / ratio)
            else:
                w = load_size
                h = int(w * ratio)
                
            input_image = input_image.resize((h, w), Image.BICUBIC)
            input_image = np.asarray(input_image)

            # RGB -> BGR
            input_image = input_image[:, :, [2, 1, 0]]
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # preprocess, (-1, 1)
            input_image = -1 + 2 * input_image
            #running mode
            input_image = Variable(input_image, volatile=True).cpu()
            #input_image = Variable(input_image, volatile=True).cuda()

            output_image = model(input_image)
            output_image = output_image[0]
            # BGR -> RGB

            output_image = output_image[[2, 1, 0], :, :]

            # deprocess, (0, 1)
            output_image = output_image.data.cpu().float() * 0.5 + 0.5
            outputfilename = os.path.join(GANIMGPATH, os.path.basename(filename)[:-4] + '_mer_' + artstyle + '.jpg')
            vutils.save_image(output_image, outputfilename)

            output_image = io.imread(outputfilename)

            t_mask = np.zeros([output_image.shape[0],output_image.shape[1]])
            for j in range(N):
                if not class_ids[j] == 1:
                    continue
                t_mask += masks[:,:,j]

            for c in range(3):
                output_image[:,:,c]=t_mask*output_image[:,:,c]

            '''
            output_image = np.array(output_image)*255
            output_image = output_image.astype('uint8')
            output_image = output_image.reshape(h,w,3)
            '''
            output_image += antimer_image
            outputfilename = os.path.join(GANIMGPATH, os.path.basename(filename)[:-4] + '_con_' + artstyle + '.jpg')
            print('output_image size:',output_image.shape[0],'*',output_image.shape[1],'\nc:',output_image.shape[2])
            io.imsave(outputfilename,output_image)


            #print(output_image.dtype)
            print('GAN_image output done')
            imgfile = io.imread(outputfilename)

            gwidth = self.ganimgshow.geometry().width()  # width of the graph , same below
            gheight = self.ganimgshow.geometry().height()
            x = imgfile.shape[1]
            y = imgfile.shape[0]
            self.zoomscale = min(gwidth / x, gheight / y)

            frame = QImage(imgfile, x, y, QImage.Format_RGB888)
            pix = QPixmap.fromImage(frame)
            
            self.item = QGraphicsPixmapItem(pix)  # 创建像素图元
            self.scene = QGraphicsScene()  # 创建场景
            self.item.setScale(self.zoomscale)
            self.scene.addItem(self.item)
            self.ganimgshow.setScene(self.scene)

            print('done')

        else:
            QMessageBox.information(self, 'error', 'no file to transform,please choose a file')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyWindow()

#----------------------signal and slot-------------------------------
    myWin.loadbutton.clicked.connect(myWin.setBrowerPath)
    myWin.openbutton.clicked.connect(myWin.readrawimage)
    myWin.tranbutton.clicked.connect(myWin.tranimage)

    myWin.show()

    sys.exit(app.exec_())
