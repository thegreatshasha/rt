from chainer import dataset
import numpy as np
import matplotlib.pyplot as plt


class SquaresDataset:
    def __init__(self):
        pass
    
    def load(self):
        pass
    
    def encode_labels(self, labels):
        """
        Encodes a list of labels as y tensor
        
        Args:
            labels: List of list of boxes
            
        Returns:
            gt_loc (n,1): 
        """
        pass
    
    def generate_batch(self, n=100, h=70, w=70, img_h=240, img_w=240):
        """
        Generates a tensor of images with n randomly places squares. Images are of size 240x240.
        
        Args:
            n (int): Batch size
            h (int): Height of each square
            w (int): Height of each square
        
        Returns:
            imgs (n, 3, 240, 240): RGB image tensor
            labels (n, v(n), 4): Numpy array with variable no of boxes per image
        """
        # image tensor
        imgs = np.zeros((n,3,240,240), dtype=np.float32)
        labels = []

        for img in imgs:
            num_boxes = 5 # np.random.randint(2,4)

            boxes = []
            for i in xrange(num_boxes):
                tx = np.random.randint(0, 239-w)
                ty = np.random.randint(0, 239-h)
                box = np.array([tx,ty,tx+w,ty+h])
                img[:,ty:ty+h,tx:tx+w] = 1
                boxes.append(box)
            
            labels.append(np.array(boxes))
        
        return imgs, np.array(labels)
    
    def visualize_batch(self, imgs, labels):
        """ 
        Takes a batch of images and labels and plots them
        
        Args:
            imgs (n, 3, 240, 240): Tensor of images
            labels: list of list of boxes/img, each box in (tx, ty, bx, by) format
            
        Returns:
            Nothing, really
        """
        
        for i, img in enumerate(imgs):
            
            boxes = labels[i]
            ax = plt.gca()
            ax.imshow(img[0])

            for box in boxes:
                ax.add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3] - box[1], fill=False, color='red'))
            plt.show()
    
    def encode(self):
        pass
    
    def get_example(self, i):
        # Here we generate and encode the 
        pass
    
""" Rotated square dataset """
""" Pawan's code """
""" Needs heavy refactoring """  
    
from PIL import Image
from PIL import ImageDraw
import math
import chainer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches, transforms 
import random 
from chainer.functions import log, hstack, huber_loss
from chainer import cuda 
import cupy 
import functions as fn 
    
    
class RotatedSquaresDataset: 
    
    def __init__(self):
        pass
    
    def rotate_point(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy



    def  get_rotated_pts(self,center, dims, angle ):
        """ What does this do? """
        # Bug here. This is messed up.
        # Which goes in which channel
        # He puts in array then rotates
        # Messy way for encoding. These results cannot be trusted.
        # Repeat encoding (deep matching prior)
        centerx, centery= center
        width, height = dims                 
        points =np.array([[centerx-(width/2),centery-(height/2)],
                    [centerx-(width/2),centery+(height/2)], 
                    [centerx+(width/2),centery+(height/2)], 
                     [centerx+(width/2),centery-(height/2)]])
        center = [centerx, centery]
        all_rotated_points = [ self.rotate_point(center,points[x,:],angle) for x in range(0,4)]
        all_rot_pts_array= np.array(all_rotated_points)
        return all_rotated_points, all_rot_pts_array


    def draw_rot_object(self,img_size, all_rotated_points):
        """ What does this do pawan? """

        img = Image.new('RGB', (img_size, img_size), (0, 0, 0))
        ImageDraw.Draw(img, 'RGBA').polygon(all_rotated_points, (1, 0, 0))
        img_array = np.array(img, dtype=np.float32)
        return img_array 


    def get_gt_loc(self, rot_array):
        """ What does this do pawan? """

        all_rot_pts_array = rot_array 

        left_ind =all_rot_pts_array[:,0].argmin()
        left_pt = all_rot_pts_array[left_ind,:]

        right_ind =all_rot_pts_array[:,0].argmax()
        right_pt = all_rot_pts_array[right_ind,:]

        return np.concatenate((left_pt,right_pt))

    def gt_image(self,numpy_image):
        """ Pawan, what does this do? """ 

        for i in range(3): 
            single_channel = numpy_image[:,:,i]

            if i==0:     
                tr_image = np.reshape(single_channel,[1,1,single_channel.shape[0], single_channel.shape[1]], order='C')
            else: 
                tr_image =np.hstack((tr_image,  np.reshape(single_channel,[1,1,single_channel.shape[0], single_channel.shape[1]], order='C')))

        return tr_image


    def show_ends(self,plot_image, rot_array, img_numpy):
        """ And what does this do? """
        rot_pt_array= rot_array
        if plot_image==True: 
            plt.figure()
            plt.imshow(img_numpy[:,:,0])
            plt.scatter(rot_pt_array[rot_pt_array[:,0].argmin(),0], rot_pt_array[rot_pt_array[:,0].argmin(),1],c=['red'])
            plt.scatter(rot_pt_array[rot_pt_array[:,0].argmax(),0], rot_pt_array[rot_pt_array[:,0].argmax(),1],c=['green'])
            plt.show()

    def generate_rot_image(self,center, dimensions, angle,img_size, plot_image=False ): 
        """
         What does this do, pawan?
         Input- 

         img_size : (2,2) array -size of the input image to draw on 

         Output- 
         box      : 1*7 numpy array [centerx, centery, width, height, angle,1,0]

        """


        all_rotated_points,rot_pt_array = self.get_rotated_pts(center,dimensions, angle )
        img_numpy  =  self.draw_rot_object(img_size, all_rotated_points)
        gt_loc     =  self.get_gt_loc(rot_pt_array) 
        rot_gt_box = gt_loc 

        #self.show_ends(plot_image,rot_pt_array, img_numpy)
        gt_img = self.gt_image(img_numpy)

        return rot_gt_box, gt_img

    def generate_batch( self, num_imgs=10, h=50, w= 30, num_boxes=2, image_size= 240):
        """
        Genreates a batch of rotated images
        
        Args:
            num_imgs (scalar): Batch size
            h (scalar): Height of the rectangle
            w (scalar): Width of the rectangle
            num_boxes (scalar): Number of boxes
            image_size (scalar): Size of the image square.
            
        Returns:
            rot_gt_imgs (num, 3, image_size, image_size)
        """

        rot_gt_box =np.zeros((num_imgs,num_boxes,4))
        rot_gt_imgs =np.zeros((num_imgs,3,image_size,image_size))
        center_range =[50,240-50]

        for simg in range(num_imgs): 

            dimensions =[h,w]

            final_image = np.zeros([1,3,image_size, image_size])

            for sbox in range(num_boxes):
                angle = (np.random.rand())*6.28
                center= [np.random.randint(center_range[0],center_range[1]),
                         np.random.randint(center_range[0],center_range[1])]
                
                s_rot_gt_loc,s_rot_img= self.generate_rot_image(center,dimensions,angle,image_size, plot_image=True  )
                rot_gt_box[simg,sbox,:]= s_rot_gt_loc
                final_image += s_rot_img


            final_image[final_image>0] =1   
            #plt.figure()
            #plt.imshow(final_image[0,0,:,:])
            #plt.show()
            rot_gt_imgs[simg,:,:,:]= final_image

        return rot_gt_imgs.astype(np.float32), rot_gt_box.astype(np.float32)
    
from skimage.transform import rescale
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.misc import imread

class BeetleDataset:
    
    def __init__(self, raw_files, label_files):
        self.raw_files = raw_files
        self.label_files = label_files
        self.raw_imgs = [imread(f, mode='RGB') for f in self.raw_files]
        self.label_imgs = [imread(f, mode='RGB') for f in self.label_files]
    
    def generate_batch(self, n=100):
        """
        Generates a tensor of images with n randomly places squares. Images are of size 240x240.
        Randomly select a file
        Extract a random patch
        DO this until batch generation is complete
        
            n (int): Batch size
        
        Returns:
            imgs (n, 3, 320, 320): RGB image tensor
            labels (n, v(n), 4): Numpy array with variable no of boxes per image
        """
        images = []
        labels = []
        
        n_count = 0

        while n_count<n:
            idx = np.random.choice(np.arange(len(self.raw_imgs)))
            
            raw_img = self.raw_imgs[idx]
            label_img = self.label_imgs[idx]
        
            patch, patch_bw = self.get_random_patch(raw_img, label_img)
            boxes = self.generate_boxes(patch_bw)
            
            # At least one label in the box
            if len(boxes):
                images.append(patch)
                labels.append(boxes)
                n_count += 1

        return np.moveaxis(np.array(images), 3, 1).astype(np.float32), np.array(labels)
        
    def get_random_patch(self, raw_img, label_img, size=160):
        # Sample centre
        cx = np.random.randint(0+size/2, raw_img.shape[1]-size/2)
        cy = np.random.randint(0+size/2, raw_img.shape[0]-size/2)
        
        # Extract and rescale patch
        patch_raw = raw_img[cy-size/2:cy+size/2,cx-size/2:cx+size/2]
        patch_label = label_img[cy-size/2:cy+size/2,cx-size/2:cx+size/2]
        
        # Upsample both raw and label patch
        patch_resized_raw = rescale(patch_raw, 2)
        patch_resized_label = rescale(patch_label, 2)
        patch_resized_bw = np.zeros((patch_resized_label.shape[0], patch_resized_label.shape[1]), dtype=np.float32)
        
        # Extract the ellipse region separately
        indices = np.where(patch_resized_label==[0,0,255])
        patch_resized_bw[indices[0],indices[1]]=1

        return patch_resized_raw, patch_resized_bw
    
    def visualize_batch(self, x, y):
        # Allows one to visualize a batch for visual verification
        # Run training first, then run visual verification for initiation
        
        for x_img, y_boxes in zip(x, y):
            fig = plt.figure(figsize=(10,10))
            # Show the img
            
            for y_box in y_boxes:
                cv2.line(x_img,(y_box[0], y_box[1]),(y_box[2], y_box[3]),(255,0,0),3)
            
            plt.imshow(x_img)
            fig.text(0.5, 0.05, str(x_img.shape), ha='center')

            plt.show()
    
    def generate_boxes(self, patch):
        # Find all contours
        contours, hierarchy = cv2.findContours(patch.astype(np.uint8),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        boxes = []

        for cnt in contours:
            # Find rotated box
            rect = cv2.minAreaRect(cnt)

            box = cv2.cv.BoxPoints(rect)
            box = np.int0(box)

            # Find left most and right most point
            p_left = box[box[:,0].argmin(),:]
            p_right = box[box[:,0].argmax(),:]

            # Add if condition to check that area of box is fine
            box_final = np.hstack([p_left, p_right]).astype(np.float32)
            boxes.append(box_final)
            #cv2.drawContours(patch_resized, [box], -1, (255,0,0),2)
            #cv2.line(patch_resized,(box_final[0], box_final[1]),(box_final[2], box_final[3]),(255,0,0),3)

        return np.array(boxes).astype(np.float32)