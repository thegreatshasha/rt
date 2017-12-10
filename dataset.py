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
            num_boxes = 2 # np.random.randint(2,4)

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