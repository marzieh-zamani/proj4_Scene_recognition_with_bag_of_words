import cv2
import numpy as np
import pickle
from utils import load_image, load_image_gray
import cyvlfeat as vlfeat
import sklearn.metrics.pairwise as sklearn_pairwise
from sklearn.svm import LinearSVC
from IPython.core.debugger import set_trace
from sklearn.metrics import confusion_matrix


def get_tiny_images(image_paths):
    """
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    To build a tiny image feature, simply resize the original image to a very
    small square resolution, e.g. 16x16. You can either resize the images to
    square while ignoring their aspect ratio or you can crop the center
    square portion out of each image. Making the tiny images zero mean and
    unit length (normalizing them) will increase performance modestly.

    Useful functions:
    -   cv2.resize
    -   use load_image(path) to load a RGB images and load_image_gray(path) to
        load grayscale images

    Args:
    -   image_paths: list of N elements containing image paths

    Returns:
    -   feats: N x d numpy array of resized and then vectorized tiny images
            e.g. if the images are resized to 16x16, d would be 256
    """
    # dummy feats variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    feat_dim = 16
    feats = np.zeros((len(image_paths), feat_dim*feat_dim))

    for i, path in enumerate(image_paths):
        img = load_image_gray(path)
        img = cv2.resize(img, (feat_dim, feat_dim)).flatten()
        img = (img - np.mean(img))/np.std(img)
        feats[i] = img

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats

def build_vocabulary(image_paths, vocab_size):
    """
    This function will sample SIFT descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
        http://www.vlfeat.org/matlab/vl_dsift.html
            -  frames is a N x 2 matrix of locations, which can be thrown away
            here (but possibly used for extra credit in get_bags_of_sifts if
            you're making a "spatial pyramid").
            -  descriptors is a N x 128 matrix of SIFT features
        Note: there are step, bin size, and smoothing parameters you can
        manipulate for dsift(). We recommend debugging with the 'fast'
        parameter. This approximate version of SIFT is about 20 times faster to
        compute. Also, be sure not to use the default value of step size. It
        will be very slow and you'll see relatively little performance gain
        from extremely dense sampling. You are welcome to use your own SIFT
        feature code! It will probably be slower, though.
    -   cluster_centers = vlfeat.kmeans.kmeans(X, K)
            http://www.vlfeat.org/matlab/vl_kmeans.html
            -  X is a N x d numpy array of sampled SIFT features, where N is
                the number of features sampled. N should be pretty large!
            -  K is the number of clusters desired (vocab_size)
                cluster_centers is a K x d matrix of cluster centers. This is
                your vocabulary.

    Args:
    -   image_paths: list of image paths.
    -   vocab_size: size of vocabulary

    Returns:
    -   vocab: This is a vocab_size x d numpy array (vocabulary). Each row is a
        cluster center / visual word
    """
    # Load images from the training set. To save computation time, you don't
    # necessarily need to sample from all images, although it would be better
    # to do so. You can randomly sample the descriptors from each image to save
    # memory and speed up the clustering. Or you can simply call vl_dsift with
    # a large step size here, but a smaller step size in get_bags_of_sifts.
    #
    # For each loaded image, get some SIFT features. You don't have to get as
    # many SIFT features as you will in get_bags_of_sift, because you're only
    # trying to get a representative sample here.
    #
    # Once you have tens of thousands of SIFT features from many training
    # images, cluster them with kmeans. The resulting centroids are now your
    # visual word vocabulary.

    dim = 128      # length of the SIFT descriptors that you are going to compute.
    vocab = np.zeros((vocab_size,dim))

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    bag_of_features=[]

    for i, path in enumerate(image_paths):
        img = load_image_gray(path)
        img = (img - np.mean(img))/np.std(img)
        _, descriptors = vlfeat.sift.dsift(img, step=[50,50], fast=True)
        bag_of_features.append(descriptors)

    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')

    vocab = vlfeat.kmeans.kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")    

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return vocab

def get_bags_of_sifts(image_paths, vocab_filename):
    """
    This feature representation is described in the handout, lecture
    materials, and Szeliski chapter 14.
    You will want to construct SIFT features here in the same way you
    did in build_vocabulary() (except for possibly changing the sampling
    rate) and then assign each local feature to its nearest cluster center
    and build a histogram indicating how many times each cluster was used.
    Don't forget to normalize the histogram, or else a larger image with more
    SIFT features will look very different from a smaller version of the same
    image.

    Useful functions:
    -   Use load_image(path) to load RGB images and load_image_gray(path) to load
            grayscale images
    -   frames, descriptors = vlfeat.sift.dsift(img)
            http://www.vlfeat.org/matlab/vl_dsift.html
        frames is a M x 2 matrix of locations, which can be thrown away here
            (but possibly used for extra credit in get_bags_of_sifts if you're
            making a "spatial pyramid").
        descriptors is a M x 128 matrix of SIFT features
            note: there are step, bin size, and smoothing parameters you can
            manipulate for dsift(). We recommend debugging with the 'fast'
            parameter. This approximate version of SIFT is about 20 times faster
            to compute. Also, be sure not to use the default value of step size.
            It will be very slow and you'll see relatively little performance
            gain from extremely dense sampling. You are welcome to use your own
            SIFT feature code! It will probably be slower, though.
    -   assignments = vlfeat.kmeans.kmeans_quantize(data, vocab)
            finds the cluster assigments for features in data
            -  data is a M x d matrix of image features
            -  vocab is the vocab_size x d matrix of cluster centers
            (vocabulary)
            -  assignments is a Mx1 array of assignments of feature vectors to
            nearest cluster centers, each element is an integer in
            [0, vocab_size)

    Args:
    -   image_paths: paths to N images
    -   vocab_filename: Path to the precomputed vocabulary.
            This function assumes that vocab_filename exists and contains an
            vocab_size x 128 ndarray 'vocab' where each row is a kmeans centroid
            or visual word. This ndarray is saved to disk rather than passed in
            as a parameter to avoid recomputing the vocabulary every run.

    Returns:
    -   image_feats: N x d matrix, where d is the dimensionality of the
            feature representation. In this case, d will equal the number of
            clusters or equivalently the number of entries in each image's
            histogram (vocab_size) below.
    """
    # load vocabulary
    with open(vocab_filename, 'rb') as f:
        vocab = pickle.load(f)

    # dummy features variable
    feats = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    for i, path in enumerate(image_paths):

        img = load_image_gray(path)
        img = (img - np.mean(img))/np.std(img)
        _, descriptors = vlfeat.sift.dsift(img, step=[9,9], fast=True)
        
        assignments = vlfeat.kmeans.kmeans_quantize(descriptors.astype(np.float32), vocab)    
        
        histo, _ = np.histogram(assignments, range(len(vocab)+1))
        if np.linalg.norm(histo) == 0:
            feats.append(histo)
        else:
            feats.append(histo / np.linalg.norm(histo))

    feats = np.array(feats)
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return feats

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats,
metric='euclidean'):
    """
    This function will predict the category for every test image by finding
    the training image with most similar features. Instead of 1 nearest
    neighbor, you can vote based on k nearest neighbors which will increase
    performance (although you need to pick a reasonable value for k).

    Useful functions:
    -   D = sklearn_pairwise.pairwise_distances(X, Y)
        computes the distance matrix D between all pairs of rows in X and Y.
            -  X is a N x d numpy array of d-dimensional features arranged along
            N rows
            -  Y is a M x d numpy array of d-dimensional features arranged along
            N rows
            -  D is a N x M numpy array where d(i, j) is the distance between row
            i of X and row j of Y

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating
            the ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    -   metric: (optional) metric to be used for nearest neighbor.
            Can be used to select different distance functions. The default
            metric, 'euclidean' is fine for tiny images. 'chi2' tends to work
            well for histograms

    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    # Define k 
    k = 16

    # Computes the distance matrix D between all pairs of test & train images
    D = sklearn_pairwise.pairwise_distances(test_image_feats, train_image_feats, 'euclidean')

    # #Find the k closest features to each test image feature
    sorted_indices = np.argsort(D, axis=1)
    knns = sorted_indices[:,0:k]

    get_labels = lambda t: train_labels[t]
    vlabels = np.vectorize(get_labels)

    # Determine the predicted_categories of those k features
    predicted_categories = np.zeros_like(knns)
    predicted_categories = vlabels(knns)

    N = test_image_feats.shape[0]
    for i in range(N):
        unique_labels, counts = np.unique(predicted_categories[i], return_counts=True)
        idx_sort = np.argsort(-counts)
        test_labels.append(unique_labels[idx_sort[0]])

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return  test_labels

def svm_classify(train_image_feats, train_labels, test_image_feats, lambd=3):
    """
    This function will train a linear SVM for every category (i.e. one vs all)
    and then use the learned linear classifiers to predict the category of
    every test image. Every test feature will be evaluated with all 15 SVMs
    and the most confident SVM will "win". Confidence, or distance from the
    margin, is W*X + B where '*' is the inner product or dot product and W and
    B are the learned hyperplane parameters.

    Useful functions:
    -   sklearn LinearSVC
        http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
    -   svm.fit(X, y)
    -   set(l)

    Args:
    -   train_image_feats:  N x d numpy array, where d is the dimensionality of
            the feature representation
    -   train_labels: N element list, where each entry is a string indicating the
            ground truth category for each training image
    -   test_image_feats: M x d numpy array, where d is the dimensionality of the
            feature representation. You can assume N = M, unless you have changed
            the starter code
    Returns:
    -   test_labels: M element list, where each entry is a string indicating the
            predicted category for each testing image
    """
    # categories
    categories = list(set(train_labels))

    # construct 1 vs all SVMs for each category
    svms = {cat: LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=lambd, max_iter=10000) for cat in categories}

    test_labels = []

    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    #############################################################################

    # Define initial 
    # print('SVM lambda : ', lambd)
    train_labels = np.array(train_labels)
    N = train_labels.shape[0]
    M = test_image_feats.shape[0]
    pred_conf = np.zeros([M,len(categories)])

    # One vs. All SVC
    for cat, svc in svms.items():        
        # train_labels_binary
        train_labels_binary = np.where(train_labels == cat,1,0)
        cat_idx = categories.index(cat)
        
        # svc fit & test
        svc.fit(train_image_feats, train_labels_binary)
        pred_conf[:,cat_idx] = svc.decision_function(test_image_feats)
    

    predicted_categories = []   
    for i in range(M):
        idx_sort = np.argsort(-pred_conf[i,:])[0]
        predicted_categories.append(categories[idx_sort])
    
    test_labels = np.copy(predicted_categories)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return test_labels

def show_results_R1(train_image_paths, test_image_paths, train_labels, test_labels,
    categories, abbr_categories, predicted_categories):
  """
  Ref : show_results function from utils
  return: prediction accuracy
  """
  cat2idx = {cat: idx for idx, cat in enumerate(categories)}

  # confusion matrix
  y_true = [cat2idx[cat] for cat in test_labels]
  y_pred = [cat2idx[cat] for cat in predicted_categories]
  cm = confusion_matrix(y_true, y_pred)
  cm = cm.astype(np.float) / cm.sum(axis=1)[:, np.newaxis]
  acc = np.mean(np.diag(cm))
  acc = int(acc*100)/100
  return acc