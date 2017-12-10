import unittest
from unittest import TestSuite
from dataset import SquaresDataset
import matplotlib.pyplot as plt
import chainer.functions as F
from skimage.transform import rotate
from scipy.ndimage import distance_transform_edt
from functions import *
from scipy.ndimage import shift


def show_image(img, title):
    """
    Plots an image with text in new figure
    
    Args:
        img: Image
        title: Text
        
    Returns:
        Nothing
    """
    fig = plt.figure()
    fig.suptitle(title, fontsize=20)
    plt.imshow(img)
    plt.show()
    
def fake_pred_data(gt_class, offset, rot):
    """ Not functional right now """
    pred = np.random.random(gt_class.data.shape).astype(np.float32)/100

    for i in range(gt_class.data.shape[0]):
        dist_img = gt_class.data[i,0,:,:]
        dist_img = distance_transform_edt(dist_img)
        dist_img = shift(dist_img, offset)
        dist_img = dist_img + rotate(dist_img, rot)
        dist_img = dist_img/dist_img.max()
        
        pred[i,0,:,:] = dist_img
        
    return chainer.Variable(pred)
    
class TestCode(unittest.TestCase):
    
    def test_dataset(self):
        """ TESTED: Test that datast works """
        # Basic plotting
        fig = plt.figure()
        fig.suptitle('Dataset generation example', fontsize=20)
        
        # Generate and visualize the dataset squares
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=20, w=20, img_h=60, img_w=60)
        db.visualize_batch(imgs, labels)
    
    def test_encode_y(self):
        """ TESTED: Manually worked out example with batch size of two. Trivial category. """
        # Encoded boxes in a couple of boxes should look like the distance transform
        fig = plt.figure()
        fig.suptitle('Encode y example', fontsize=20)
        
        # Let's try with random values first
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=40, w=80, img_h=240, img_w=240)
        gt_class, gt_loc = encode_y(imgs, labels)

        # Plot the encoded values
        show_image(gt_loc[0,0,:,:].data, 'Encoding tx channel')
        show_image(gt_loc[0,1,:,:].data, 'Encoding ty channel')
        show_image(gt_loc[0,2,:,:].data, 'Encoding bx channel')
        show_image(gt_loc[0,3,:,:].data, 'Encoding by channel')
        
    def test_downsample(self):
        """ LOGIC: Test that shape matches. """
        # Basic plotting
        fig = plt.figure()
        fig.suptitle('Downsampling visual test', fontsize=20)
        
        # Let's generate some boxes
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=80, w=80, img_h=240, img_w=240)
        db.visualize_batch(imgs, labels)
        
        # Check that boxes redrawn after downsampling look identical to original ones
        imgs_down, labels_down = downsample(imgs, labels)
        db.visualize_batch(imgs_down, labels_down)
    
    def test_predict(self):
        """ LOGIC: What does this method do? """
        # Visualizing should also be fine
        
        # Do not implement for now. Only get the regression loss to work.
        pass
    
    def test_network(self):
        """ LOGIC: Test that the shape is correct? """
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=20, w=20, img_h=240, img_w=240)
        
        vgg = VGGNet()
        
        pred_loc, pred_class= vgg(imgs)
        print(' pred_loc shape :', pred_loc.shape)
        print(' pred_class shape :', pred_class.shape)
        
        # Need to add more tests here. Just checking for shape is not sufficient.
    
    def test_loss(self):
        """ LOGIC: Calculate sample loss for the network. Trivial category. """
        print(loss(pred_class, pred_loc, gt_class, gt_loc, lambd=0.4))
        
        # Not really needed but let's still add a debug flag and check that this works
        pass
    
    def test_reg_loss(self):
        """ LOGIC: Calculate regression loss for the network with batch size of two. Trivial category. """
        # Shift by two pixels and check?
        # Basic plotting
        fig = plt.figure()
        fig.suptitle('Regression loss example', fontsize=20)
        
        # Should we show the regression loss hm here?
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=20, w=20, img_h=240, img_w=240)
        x_down, labels_down = downsample(imgs, labels)
        
        # Loss calculation
        pred_class, pred_loc = vgg(imgs)
        gt_class, gt_loc = encode_y(imgs, labels)
        reg_loss = regression_loss(pred_loc, gt_loc, gt_class)
        
        # Plot and print the final regression loss
    
    def test_class_loss(self):
        """ LOGIC: Calculate classification loss for the network with batch size of two. Trivial category. """
        # Should we show the classification loss heatmap here?
        # Shift by two pixels and check?
        # Generate data
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=40, w=40, img_h=240, img_w=240)
        imgs_down, labels_down = downsample(imgs, labels)
        gt_class, gt_loc = encode_y(imgs_down, labels_down)
        
        # Loss calculation        
        configs = [(0,0,0), (5,0,5), (10,0,10), (15,0,15)]
        
        # Loop over various offset x,y,angles and generate loss
        for delx, dely, angle in configs:
            pred_class = fake_pred_data(gt_class, (dely, delx), angle)

            class_loss = classification_loss(pred_class, gt_class, debug=True)
            abs_loss = (pred_class - gt_class) ** 2
            #plt.show()

            show_image(gt_class[0,0,:,:].data, 'Ground truth_offset_%d_%d_angle_%d'%(delx, dely, angle))
            show_image(pred_class[0,0,:,:].data, 'VGG prediction_offset_%d_%d_angle_%d'%(delx, dely, angle))
            show_image(abs_loss[0,0,:,:].data, 'Absolute class loss_offset_%d_%d_angle_%d'%(delx, dely, angle))
            show_image(class_loss[0,0,:,:].data, 'Classification loss_offset_%d_%d_angle_%d'%(delx, dely, angle))
    
    def test_classifier_learning_capacity(self):
        """ Is the network able to do any auxillary learning task properly """
        pass
    
    def test_convergence(self):
        """ Check that for the one and two squares datasets, the network loss convergence. Plot the loss and show that it converges. """
        pass
    
    def test_network_normalized(self):
        """ Check that the network outputs normalized probabilities """
        pass
    
    def test_autoencoder(self):
        """ Check that learning works on the task with simple examples """
    
    def test_select_mask(self):
        """ For controlled inputs, check what the mask looks like. Does it look sensible? """
        # Plotting stuff
        fig = plt.figure()
        fig.suptitle('Selection mask visualization', fontsize=20)
        
        # Generate data
        db = SquaresDataset()
        imgs, labels = db.generate_batch(n=2, h=40, w=60, img_h=240, img_w=240)
        imgs_down, labels_down = downsample(imgs, labels)
        gt_class, gt_loc = encode_y(imgs_down, labels_down)

        #vgg = VGGNet()
        # Shifts in x, y, angle   
        configs = [(0,0,0), (5,0,15), (10,0,30), (15,0,45)]
        
        # Loop over various offset x,y,angles and generate loss
        for delx, dely, angle in configs:
            pred_class = fake_pred_data(gt_class, (dely, delx), angle)

            abs_loss = (pred_class - gt_class) ** 2
            mask = selection_mask(abs_loss.data, gt_class.data)

            show_image(gt_class[0,0,:,:].data, 'Ground truth_offset_%d_%d_angle_%d'%(delx, dely, angle))
            show_image(pred_class[0,0,:,:].data, 'VGG prediction_offset_%d_%d_angle_%d'%(delx, dely, angle))
            show_image(mask[0,0,:,:], 'Selection_mask_offset_%d_%d_angle_%d'%(delx, dely, angle))
            show_image(mask[0,0,:,:]-gt_class[0,0,:,:].data, 'Negative_mask_offset_%d_%d_angle_%d'%(delx, dely, angle))