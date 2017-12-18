import numpy as np
from network import VGGNet
import chainer
import chainer.functions as F
from chainer import cuda

def encode_y(x_down, labels_down):
    """
    x_downsampled tensor -> y_tensor: Numpy
    
    Args:
        x_down (b, 3, 60, 60): Downsampled list of images
        labels_down (b, v, 2): List of lists of lists (downsampled boxes)
        dimensions(2,1)      : Contains the height and width of the object(height width of rectangle) 
    Returns:
        y_class (b, 1, 60, 60): Chainer variable containing mask for each image. Calculated within encode
        y_loc (b, 4, 60, 60): Chainer variable containing offsets
    """
    
    y_class = np.zeros((x_down.shape[0], x_down.shape[2], x_down.shape[3]), dtype=np.float32)
    y_loc = np.zeros((x_down.shape[0], 8, x_down.shape[2], x_down.shape[3]), dtype=np.float32)
    
    y_class[x_down[:,2,:,:]>0] = 1 # Can also choose a smaller neighbourhood here
    pos_inds = np.argwhere(y_class) # y_class is also the positive examples mask
    
    for b, y, x in pos_inds:
        y_loc[b, :, y, x] = match_boxes(x, y, labels_down[b])
        
    y_class = y_class.reshape(x_down.shape[0], 1, x_down.shape[2], x_down.shape[3])
   # y_loc[:,0,:,:] = y_loc[:,0,:,:]*-1.0
    #y_loc[:,1,:,:] = y_loc[:,1,:,:]*-1.0
    
    return chainer.Variable(y_class), chainer.Variable(y_loc)

def match_boxes(x, y, boxes):
    """ Numpy
    Matches a point (x,y) to a bunch of boxes. Returns offset of the box with the nearest centre.
    
    Args:
        x (scalar): X coordinate of point being matched
        y (scalar): Y coordinate of point being matched
        boxes: List of list of downsampled boxes being matched, in (tx, ty, bx, by) notation
    
    """
    
    dist = 10**5 # Store the smallest disance to a large no initially, can glitch if dist greater than this
    
    for box in boxes:
        cx = (box[0] + box[2])/2
        cy = (box[1] + box[3])/2
        
        box_dist = np.sqrt((cx - x)**2 + (cy - y)**2)
        
        if box_dist < dist:
            offset = np.array([(box[0] - x),(box[1] - y),
                                      (box[2] - x),(box[3] - y),
                                      (box[4] - x),(box[5] - y),
                                      (box[6] - x),(box[7] - y),
                                     ])/12.25 # offset is still distance
            dist = box_dist
            
    # Should not glitch because matching is only done for positive indices
    return offset
    

def downsample(x, labels):
    """
    x -> x/4
    
    Args:
        x (b, 3, 240, 240): Batch of 3 channel 240 x 240 images
        labels (b, v, 4): List of list of boxes

    Returns:
        x_down (b, 3, 60, 60): Batch of 3 channel 60x60 images
        labels_down (b, v, 4): List of list of downsampled boxes
    """
    # Resize batch by two max pools
    x_down = F.max_pooling_2d(F.max_pooling_2d(x, 2, stride=2 ), 2, stride=2 ) # Is this correct? Should we use bilinear interpolation instead?
    labels_down = labels/4.0
    
    return x_down.data, labels_down

def loss(pred_class, pred_loc, gt_class, gt_loc, lambd=1):
    """
    Calculates weighted sum of classification and regression loss. Calls the classification loss and regression loss functions separately.
    
    Args:
        pred_class (b, 1, 60, 60): Network confidence probs for images
        pred_loc (b, 4, 60, 60): Network offsets for each location
        gt_class (b, 1, 60, 60): Gt class scores from encode
        gt_loc (b, 4, 60, 60): Gt regression offsets from encode
        lambd (scalar): WEighting factor comparison regression loss to 
        
    Returns:
        loss: Scalar value of 
    """
    return classification_loss(pred_class, gt_class) + lambd * regression_loss(pred_loc, gt_loc, gt_class)

def classification_loss(pred_class, gt_class, debug=False):
    """
    Classification loss from mean squared diff between probabilities. Should probably use cross entropy instead but usng this now for simplicity.
    
    Also does hard negative mining. so requires generation of a selction mask of positives and most overconfident negatives.
    
    Args:
        pred_class (b, 1, 60, 60): Network confidence probs (on gpu)
        gt_class (b, 1, 60, 60): Binary gt confidence probs (on gpu)
        
    Returns:
        class_loss: Scalar (on gpu)
    """
    abs_loss = (pred_class - gt_class) ** 2
    mask = selection_mask(cuda.to_cpu(abs_loss.data), cuda.to_cpu(gt_class.data))
    selected_loss = abs_loss * cuda.to_gpu(mask)
    
    if debug:
        return selected_loss/pred_class.shape[0]
    else:
        return F.mean(selected_loss)/pred_class.shape[0]

def regression_loss(pred_loc, gt_loc, gt_class, debug=False): #PROBLEMATIC
    """
    Regression loss from vanilla mean squared diff between shifts.
    
    Args:
        pred_loc (b, 4, 60, 60): Network offsets for top left and bottom right of box. Should 
        gt_loc (b, 4, 60, 60): Ground truth offsets for top left and bottom right of box.  
        gt_class (b, 1, 60, 60): Offsets for positive examples
        
    Returns:
        reg_loss: Scalar
    """
    abs_loss = F.mean(((pred_loc - gt_loc) ** 2),axis=1) # Check dims in test
    abs_loss = F.reshape(abs_loss,(abs_loss.shape[0],1,abs_loss.shape[1], abs_loss.shape[2]))
    selected_loss = abs_loss * gt_class
    
    if debug:
        return selected_loss/pred_loc.shape[0]
    else:
        return F.mean(selected_loss)/pred_loc.shape[0]

def selection_mask(abs_loss, gt_class): #SAFE
    """ Is there a simpler way of doing this?
    Returns a binary mask from absolute mean square classification loss and the ground truth mask
    
    Args:
        abs_loss (b, 1, 60, 60): Absolute probability loss value over each pixel
        gt_class (b, 1, 60, 60): Binary gt probs to set the positive pixels to one
        
    Returns;
        select_mask (b, 1, 60, 60): Selection mask for poth positive and negative pixels0
    """
    yinv = 1 - gt_class
    
    loss_neg = yinv*abs_loss

    select_mask = np.zeros(gt_class.shape)

    for num,i in enumerate(loss_neg):
        # Getting the sorted indices loss in a flat array
        indices = np.argsort(i,axis=None )

        # Reshaping the flat indices to matrix indices
        matrix_indices = np.unravel_index(indices,(abs_loss.shape[2], abs_loss.shape[3]))
        matrix_indices_flipped = np.fliplr(matrix_indices)

        num_positives = int(np.sum(gt_class[num,0,:,:]))
        
        # Taking the top num_positives indices only
        matrix_indices_sorted = matrix_indices_flipped[:,0:num_positives]

        select_mask[num,0,matrix_indices_sorted[0],matrix_indices_sorted[1]] = 1 # Setting the top negative indices to one
        select_mask[num,0,:,:] = select_mask[num,0,:,:] + gt_class[num,0,:,:] # Setting all positive indices to one
        
    return select_mask

""" Some utility functions here """

# Let's try generating a rotated rectangle first. How do we do it?
def rotate_point(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)

        return qx, qy
   
import math
import cv2
from scipy.spatial import ConvexHull

def draw_quad(img, quad):
    cv2.line(img, tuple(quad[0]), tuple(quad[1]), color=(255,0,0))
    cv2.line(img, tuple(quad[1]), tuple(quad[2]), color=(0,255,0))
    cv2.line(img, tuple(quad[2]), tuple(quad[3]), color=(0,0,255))
    cv2.line(img, tuple(quad[3]), tuple(quad[0]), color=(255,255,0))

def slope(p1, p2):
    """
    Calculates slope of two 2x1 points
    
    Args:
        p1 (2,1): x1, y1
        p2 (2,1): x2, y2
        
    Returns:
        slope (scalar): Slope of the line between the two points
    """
    if p2[0] == p1[0]:
        return np.inf
    else:
        return float(p2[1]-p1[1])/float(p2[0]-p1[0])

def order_pts(pts):
    """
    Takes a list of pts and finds a structured labelling to it
    
    Args:
        pts (4,2): 4x2 list of unordered pts
        
    Returns:
        pts_u (4,2): 4x2 list of ordered pts
    """
    pts_u = pts.copy()
    hull = ConvexHull(pts)
    pts_u = hull.points[hull.vertices]
    
    # This gives us diametrically opposite points
    diag1 = pts_u[0::2]
    diag2 = pts_u[1::2]
    
    diags = [diag1, diag2]
    
    diag_idsx = np.argmax([slope(*list(diag1)), slope(*list(diag2))])
    
    x_idsx = diags[diag_idsx][:,0].argmin()
    
    min_x_1 = diags[diag_idsx][x_idsx]
    
    while not np.array_equal(pts_u[0], min_x_1):
        pts_u = np.roll(pts_u, 1, axis=0)
    
    return pts_u

def random_quadilateral():
    verts = []
    
    for i in xrange(4):
        verts.append([np.random.randint(0,400),np.random.randint(0,400)])
        
    return np.array(verts)
    #return np.array([[0,0], [100,100], [100,0], [0,100]]) + 100