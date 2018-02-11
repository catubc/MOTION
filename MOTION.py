import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import cv2, os, sys, glob
import scipy
import sklearn
import imageio
import matplotlib.cm as cm
import matplotlib
import time

from sklearn import decomposition, metrics, manifold, svm
from tsne import bh_sne
from matplotlib.path import Path
from numpy import linalg as LA
from scipy.signal import butter, filtfilt, cheby1
from scipy.spatial import distance

#**************************************************************************************************************************
#*************************************************CODE START***************************************************************
#**************************************************************************************************************************
class MOTION(object):
    ''' Class to detect motion in behaviour video;
        self.crop() to select only part of video (speeds up analysis)
        self.dimreduce() to reduce dimensionality of video and increase SNR
        self.detect_motion() compute euclidean distance between frames and plots timecourse
    '''
    
    def __init__(self,filename):
        print "...current session: ", filename
        self.filename = filename


    def crop(self):
	''' Function crops the FOV for image registration (stable region) and area of interest
	    Also converts .avi -> .npy format for both stacks + entire original movie.
	    Currently only rGb channel saved; possibly might improve to average all
	'''
        #**************** SELECT CROPPED AREA TO TRACK MOTION (smaller is faster) **********************
        #Load and save to disk # frames, frame rate and sample frame for cropping 
	if os.path.exists(os.path.split(self.filename)[0]+"/nframes.txt")==False:
	    camera = cv2.VideoCapture(self.filename)
	    self.frame_rate = camera.get(5)
	    ctr=0
	    print "reading frames"
	    while True:
		print (ctr)
		(grabbed, frame) = camera.read()
		if not grabbed: break
		ctr+=1
		if ctr==100: 
		    image_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		    np.save(os.path.split(self.filename)[0]+"/original_image.npy", image_original)
	    self.n_frames=ctr
	    np.savetxt(os.path.split(self.filename)[0]+"/nframes.txt",[self.n_frames])
	    np.savetxt(os.path.split(self.filename)[0]+"/frame_rate.txt",[self.frame_rate])
	    cv2.destroyAllWindows()
	    camera.release()
	else:
	    image_original = np.load(os.path.split(self.filename)[0]+"/original_image.npy")
	    self.n_frames = np.loadtxt(os.path.split(self.filename)[0]+"/nframes.txt",dtype='int32')
	    self.frame_rate = np.loadtxt(os.path.split(self.filename)[0]+"/frame_rate.txt",dtype='float32')
        
	self.frame_xwidth = len(image_original); self.frame_ywidth = len(image_original[0])
        #Run cropping functions on sample frame
	self.crop_frame_box(image_original, motion_correct_flag=True)      #DEFINE BOX AREAS FOR CROPPING; first define area for register
	self.crop_frame_box(image_original, motion_correct_flag=False)      #DEFINE BOX AREAS FOR CROPPING; first define area for register

	#Convert original file and cropped to .npy 
	crop_registry = np.load(self.filename[:-4]+'_registry_cropped.npz')
	self.x1_registry = crop_registry['x1']; self.x2_registry = crop_registry['x2']
	self.y1_registry = crop_registry['y1']; self.y2_registry = crop_registry['y2']
	
	crop_area = np.load(self.filename[:-4]+'_'+self.area+'_cropped.npz')
	self.x1 = crop_area['x1']; self.x2 = crop_area['x2'] 
	self.y1 = crop_area['y1']; self.y2 = crop_area['y2']
	if os.path.exists(self.filename[:-4]+'_'+self.area+'_cropped.npy')==False:
	    print "... converting .avi -> .npy files (only Green channel) ..."
	    if os.path.exists(self.filename[:-4]+'.npy')==False:
		original_frames = np.zeros((self.n_frames, self.frame_xwidth,self.frame_ywidth),dtype=np.uint8)
	    cropped_frames = np.zeros((self.n_frames, self.x2-self.x1,self.y2-self.y1),dtype=np.uint8)
	    registry_frames = np.zeros((self.n_frames, self.x2_registry-self.x1_registry,self.y2_registry-self.y1_registry),dtype=np.uint8)
	    camera = cv2.VideoCapture(self.filename)
	    ctr = 0
	    while True:
		if ctr%1000==0: print " loading frame: ", ctr
		if 'luis' in self.filename: 
		    if ctr>15000: 
			print "...************ too many frames, exiting on 15000..."
			break
		(grabbed, frame) = camera.read()
		if not grabbed: break
		
		#Save copy of frame for .npy file
		if os.path.exists(self.filename[:-4]+'.npy')==False:
		    original_frames[ctr]=frame[:,:,1]		#Save green ch only
		#original_frames.append(np.uint8(np.mean(frame, axis=2)))		#Save average of RGB chans
		
		#Crop FOV for analysis
		cropped_frames[ctr]=frame[:,:,1][self.x1:self.x2,self.y1:self.y2]
		#cropped_frames.append(np.uint8(np.mean(frame[self.x1:self.x2,self.y1:self.y2],axis=2)))
		
		#Crop FOV for registry
		registry_frames[ctr]=frame[:,:,1][self.x1_registry:self.x2_registry,self.y1_registry:self.y2_registry]
		#registry_frames.append(np.uint8(np.mean(frame[self.x1_registry:self.x2_registry,self.y1_registry:self.y2_registry],axis=2)))
		
		ctr+=1

	    #Save original movie in .npy format
	    if os.path.exists(self.filename[:-4]+'.npy')==False:
		np.save(self.filename[:-4]+'.npy', original_frames)    			#This is the entire movie converted to .npy
	    #Save cropped movie area and registry area 
	    np.save(self.filename[:-4]+'_'+self.area+'_cropped', cropped_frames)	#just cropped movie
	    np.save(self.filename[:-4]+'_registry_cropped', registry_frames)	#just cropped movie
	    

    def binarize_frames(self):
        ''' Reduce the size/dimensionality of the sample/frame by calling various functions 
            This also binarizes the frames (i.e. all vals are 0/1
            TODO: Check this step, investigate if preserving more information in the would be hefpful 
        '''
	
	#area_filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+".npy"
	#area_filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+".npy"
	area_filename = self.filename[:-4]+"_"+self.area+"_cropped_registered_"+self.mode+".npy"

	self.movie_filename = self.filename[:-4]+'.npy'
        if os.path.exists(area_filename)==False:
	    frames = np.load(self.filename[:-4]+"_"+self.area+"_cropped_registered.npy")
	    rLow=100; rHigh=255
            reduced_frames = []
	    contour_frames = []
	    edge_frames = []
            frame_count = 0
	    for frame in frames:
                if (frame_count%1000)==0: print " reducing frame: ", frame_count
                
		#Crop frame before processing
		#frame = frame[self.x1:self.x2,self.y1:self.y2]
		
		if self.mode=='all':
		    reduced_frames.append(self.decimate(frame, frame_count, rLow, rHigh))
		elif self.mode == 'contours':
		    contour_frames.append(self.find_contours(frame, frame_count, rLow, rHigh))
                elif self.mode=='edges':
		    edge_frames.append(self.find_edges(frame, frame_count, rLow, rHigh))
	    
		frame_count += 1
	    
	    cv2.waitKey(1)
	    cv2.destroyAllWindows()
	    cv2.waitKey(1)


	    if self.mode=='all': 
		np.save(area_filename, np.nan_to_num(reduced_frames))
		self.decimated_frames = np.nan_to_num(reduced_frames)
	    elif self.mode=='contours':
		np.save(area_filename, np.nan_to_num(contour_frames))
		self.decimated_frames = np.nan_to_num(contour_frames)		
	    elif self.mode=='edges': 
		np.save(area_filename, np.nan_to_num(edge_frames))
		self.decimated_frames = np.nan_to_num(edge_frames)
	    
	else:
            self.decimated_frames = np.load(area_filename,mmap_mode='c')


    def detect_movement(self):
	''' Detect movement as euclidean distance between frames
	'''
        print "... detecting movement ..."

        if os.path.exists(self.filename[:-4]+"_diff_array.npy")==False:
            self.compute_diff()

	    #Plot data 
	    t = np.arange(len(self.diff_array))/(self.frame_rate)
	    plt.plot(t, self.diff_array)

	    #Plotting parameters
	    plt.xlim(0,t[-1])
	    plt.yticks([])
	    font_size = 20
	    plt.xlabel("Time (sec)", fontsize = font_size)
	    plt.ylabel("Movement index (a.u.)", fontsize = font_size)
	    plt.tick_params(axis='both', which='both', labelsize=font_size)
	    plt.title(self.filename, fontsize = font_size)
	    plt.show(block=True)
        
	else:
            self.diff_array = np.load(self.filename[:-4]+"_diff_array.npy")


    def read_metadata(self, output):

	decimated_filename = self.filename[:-4]+"_"+self.area+"_cropped_registered_"+self.mode+".npy"
	n_frames = len(np.load(decimated_filename,mmap_mode='c'))

	names = np.loadtxt(self.filename[:-4]+"_"+self.area+"_"+str(self.methods[self.method])+"_clusternames.txt",dtype='str')
	print names
	indexes = np.load(self.filename[:-4]+"_"+self.area+"_"+str(self.methods[self.method])+"_clusterindexes.npy")
	
	#Licking
	idx = np.where(names=='lick')[0]
	if len(idx)!=0: output.lick_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=len(indexes[idx][0])/float(n_frames)
	else: output.lick_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=0
	
	#Pawing
	idx = np.where(names=='paw')[0]
	if len(idx)!=0: output.paw_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=len(indexes[idx][0])/float(n_frames)
	else: output.paw_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=0
	    
	#Add scratching to pawing 
	idx = np.where(names=='scratch')[0]
	if len(idx)!=0: output.scratch_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=len(indexes[idx][0])/float(n_frames)
	else: output.scratch_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=0
	data = np.load(glob.glob(os.path.split(self.filename)[0]+'/*_metadata.npz')[0])
	if data['drift']=='y': self.drift=1
	elif data['drift']=='n': self.drift=0
	else: print "...exception..."; quit()
	
	if data['spout_moved']=='y': self.spout_moved=1
	elif data['spout_moved']=='n': self.spout_moved=0
	else: print "...exception..."; quit()

	if data['hand_inview']=='y': self.hand_inview=1
	elif data['hand_inview']=='n': self.hand_inview=0
	else: print "...exception..."; quit()

	if data['camera_moved']=='y': self.camera_moved=1
	elif data['camera_moved']=='n': self.camera_moved=0
	else: print "...exception..."; quit()

	output.drift_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=self.drift
	output.spout_matrix[self.rt_ctr*output.scale:(self.rt_ctr+1)*output.scale,self.ses_ctr*output.scale:(self.ses_ctr+1)*output.scale]=self.spout_moved

	self.other_exclusion=data['other_exclusion']

	return output

    def load_frames(self, cluster_name):
	
	names = np.loadtxt(self.filename[:-4]+"_"+self.area+"_"+str(self.methods[self.method])+"_clusternames.txt",dtype='str')
	print names
	indexes = np.load(self.filename[:-4]+"_"+self.area+"_"+str(self.methods[self.method])+"_clusterindexes.npy")
	cluster_index = np.where(names ==cluster_name)[0]
	
	if len(cluster_index)==0: 
	    return None
	cluster_indexes = indexes[cluster_index][0]	#List of indexes for selected behaviour
	
	#Load movie 
	self.movie_filename = self.filename[:-4]+'.npy'
	enlarge = 100	#Make movie FOV larger than original cropping rectangle by 50pixels or so; otherwies difficult to see what's going on; 
        movie_array = np.load(self.movie_filename, mmap_mode='c')[:, max(0,self.x1-enlarge):self.x2+enlarge, max(0,self.y1-enlarge):self.y2+enlarge]
	print movie_array.shape
        
	if len(cluster_index)==0: 
	    return movie_array[0]*0

	#Randomly return one of these images;
	return movie_array[np.random.choice(cluster_indexes)]


    def save_metadata(self):
	
	print self.filename[:-4]
	metadata = []
	drift = raw_input("Did camera drift ? (y/n) " )
	spout_moved = raw_input("Was spout readjusted ? (y/n) " )
	hand_inview = raw_input("Did hand enter the screen ? (y/n) " )
	camera_moved = raw_input("Did camera move or otherwise jump ? (y/n) " )
	other_exclusion = raw_input("Any thing else to note (y/n or type out) ")
	
	np.savez(self.filename[:-4]+"_metadata.npz", drift=drift, spout_moved=spout_moved, hand_inview=hand_inview, camera_moved=camera_moved, other_exclusion=other_exclusion)
	

    def annotate_frames(self):
        ''' Function to annotate frames in partially supervised fashion
            Calls mupltiple functions
        '''
        
        #Subsample frames to further reduce dimensionality and speed up processing
        if True: self.subsample_frames()
	else: self.data_subsampled = self.decimated_frames

	#Scale the frame information by some coefficient of movement
	if False: self.scale_moving_frames()	

        #Run dim reduction
        self.dimension_reduction()
        
        #Filter transformed distributions to remove camera drift (usually)
        if True:
            self.filter_PCA(self.data_dim_reduction, filtering=True, plotting=True)
        
        #Cluster data
        self.cluster_methods = ['KMeans', 'MeanShift', 'DBSCAN', 'manual']
        self.cluster_method = 3
        self.cluster_data()
        
        #Review clusters and re/cut them
        #self.review_clusters()
	self.export_clusters(recluster_flag=False)
    

    def resplit_cluster(self, cluster):
        ''' Recluster previously split clusters...
        '''         
        print "... resplitting cluster: ", cluster
	
	#THIS NEEDS TO BE SIMPLIFIED
	#Subsample frames to further reduce dimensionality and speed up processing
        if True: self.subsample_frames()
	else: self.data_subsampled = self.decimated_frames

	#Scale the frame information by some coefficient of movement
	if False: self.scale_moving_frames()	

        #Run dim reduction
        self.dimension_reduction()
        
        #Filter transformed distributions to remove camera drift (usually)
        if True:
            self.filter_PCA(self.data_dim_reduction, filtering=True, plotting=False)
	self.load_clusters()
	
        
	#Load clustered info
	cluster_names = np.loadtxt(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusternames.txt", dtype='str')
	cluster_indexes = np.load(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusterindexes.npy")

	#Assign clusters to unique ids
	cumulative_indexes=[]
	unique_names = np.unique(self.cluster_names)
	print self.cluster_names
	print unique_names
	
	unique_indexes = []
	for ctr1, unique_name in enumerate(unique_names):
	    unique_indexes.append([])
	    for ctr, cluster_name in enumerate(self.cluster_names):
		if unique_name==cluster_name:
		    unique_indexes[ctr1].extend(self.cluster_indexes[ctr])
    
	cluster_id = np.where(unique_names==cluster)[0]
	print "... cluster_id: ", cluster_id
	self.unique_indexes = unique_indexes[cluster_id]

        #Cluster data
        self.cluster_methods = ['KMeans', 'MeanShift', 'DBSCAN', 'manual']
        self.cluster_method = 3
        self.cluster_data(indexes=unique_indexes[cluster_id])		#Send indexes for the selected cluster after collapsing over unique valus
        

    def resave_clusters(self,indexes):
	''' Load original cluster labels and re-adjust based on resplit cluster
	'''
	
	reclustered_id_indexes = np.int32(indexes)
	print "... reclustered id indexes: ", len(reclustered_id_indexes)
	
	#Load clustered info
	original_cluster_names = np.loadtxt(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusternames.txt", dtype='str')
	original_cluster_indexes = np.load(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusterindexes.npy")

	#Delete the cluster that was just resplit
	temp_index = np.where(original_cluster_names==self.recluster_id)[0]
	#print "... reclustered id : ", temp_index
	
	original_cluster_names = np.delete(original_cluster_names, temp_index,0)
	original_cluster_indexes = np.delete(original_cluster_indexes, temp_index,0)
	
	#Append new labels back in; first convert to lists, easier to work with due to variable length
	cluster_names_array = []
	for k in range(len(original_cluster_names)): 
	    cluster_names_array.append(original_cluster_names[k])
	
	#Add new labels back in from newly identified self.cluster_names
	for k in range(len(self.cluster_names)):
	    cluster_names_array.append(self.cluster_names[k])	
	self.cluster_names = cluster_names_array
	
	#Do the same with cluster indexes  
	cluster_indexes_array = []
	for k in range(len(original_cluster_indexes)): 
	    cluster_indexes_array.append(original_cluster_indexes[k])
	
	#Add new labels back in            ******************* NOTE: Indexes will be relative to the previously clustered indexes not 0
	for k in range(len(self.cluster_indexes)):
	    print k, len(self.cluster_indexes[k]), len(reclustered_id_indexes)
	    print self.cluster_indexes[k]
	    cluster_indexes_array.append(reclustered_id_indexes[np.int32(self.cluster_indexes[k])])	
	self.cluster_indexes = cluster_indexes_array
	
	print ".... check that all frames have been saved..."
	print len(self.cluster_indexes)
	#print np.unique(np.array(self.cluster_indexes))


	#*****Re-assign clusters to unique ids after adding the new split cluster labels back in
	cumulative_indexes=[]
	unique_names = np.unique(self.cluster_names)
	print "...reclustered data..."
	print self.cluster_names
	for k in range(len(self.cluster_indexes)):
	    print len(self.cluster_indexes[k])
	print "\n\n... unique data..."
	print unique_names
	
	unique_indexes = []
	for ctr1, unique_name in enumerate(unique_names):
	    unique_indexes.append([])
	    for ctr, cluster_name in enumerate(self.cluster_names):
		if unique_name==cluster_name:
		    unique_indexes[ctr1].extend(self.cluster_indexes[ctr])
	    print len(unique_indexes[ctr1])
	#cluster_id = np.where(unique_names==cluster)[0]
	#print "... cluster_id: ", cluster_id

	np.savetxt(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusternames_new.txt", unique_names,fmt='%s')
	np.save(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusterindexes_new.npy", unique_indexes)

	self.export_clusters(recluster_flag=True)


    def manual_label(self):
	
	filename_manuallabel = self.filename[:-4]+"_"+self.area+"_manuallabels.npz"
	
	if os.path.exists(filename_manuallabel)==False:
	    
	    plt.plot(self.diff_array)
	    mean = np.mean(self.diff_array)
	    top_cutoff = np.max(self.diff_array)*.55
	    bottom_cutoff = np.mean(self.diff_array)*.05
	    plt.plot([0,len(self.diff_array)],[top_cutoff,top_cutoff])
	    plt.plot([0,len(self.diff_array)],[bottom_cutoff,bottom_cutoff])
	    plt.show(block=True)
	    
	    print "... limitting annotation to 50 events max..."
	    indexes = np.where(self.diff_array>top_cutoff)[0]
	    indexes = indexes[np.random.randint(len(indexes),size=50)]
	    print "... # frames: ", len(indexes)
	    
	    indexes2 = np.where(self.diff_array<bottom_cutoff)[0]
	    indexes2 = indexes2[np.random.randint(len(indexes2),size=50)]

	    print "... # frames: ", len(indexes2)
	    indexes = np.hstack((indexes,indexes2))
	    print "... # total frames to annotate: ", len(indexes)
	    enlarge=100
	    movie_array = np.load(self.movie_filename, mmap_mode='c')[:, max(0,self.x1-enlarge):self.x2+enlarge, max(0,self.y1-enlarge):self.y2+enlarge]

	    #Select most active frames
	    data_ = movie_array[indexes]
	    border = 30
	    fontsize=15
	    classifier = np.array([0,0,0,0])
	    classifier_list = []
	    self.complete=False
	    for k,frame in enumerate(data_):
		if self.complete==True: break		#Exit by clicking outside the annotation box
		
		#Make nice box around each frame to use for annotation
		temp = np.zeros((frame.shape[0]+border,frame.shape[1]+border))
		temp[:, :border/2]=100
		temp[:, frame.shape[1]+border/2:]=150
		temp[:border/2]=50
		temp[frame.shape[0]+border/2:]=200
		
		temp[border/2:frame.shape[0]+border/2,border/2:frame.shape[1]+border/2]=frame[:,:,1]
		
		#Make plots
		fig, ax = plt.subplots()	    
		ax.imshow(temp)

		self.cid = fig.canvas.mpl_connect('button_press_event', self.on_click_classify)
		plt.suptitle("frame: "+str(k)+" / "+str(len(data_)),fontsize=fontsize)
		plt.title("Lick: "+str(classifier[0]),fontsize=fontsize)
		plt.xlabel("Stationary: "+str(classifier[1]),fontsize=fontsize)
		plt.ylabel("Paw :"+str(classifier[2]),fontsize=fontsize)
		plt.show(block=True)
		
		y = self.coords[0] 
		x = self.coords[1]
		if (y<(border/2)): 
		    classifier[0]+=1
		    classifier_list.append(0)
		elif (y>(frame.shape[0]+border/2)): 
		    classifier[1]+=1
		    classifier_list.append(1)
		elif (x<(border/2)): 
		    classifier[2]+=1
		    classifier_list.append(2)
		else: 
		    classifier[3]+=1
		    classifier_list.append(3)
	    
	    np.savez(self.filename[:-4]+"_"+self.area+"_manuallabels",indexes=indexes[:k],classification=classifier_list[:k])

    def classify_frames(self):

	self.manual_label()

	#Load classified data
	data = np.load(self.filename[:-4]+"_"+self.area+"_manuallabels.npz")
	indexes=data['indexes']
	class_id=data['classification']
	
	#if True: self.subsample_frames()
	if True: 
	    self.dimension_reduction()
	    self.data = self.data_dim_reduction

	if True: self.filter_data(self.data_dim_reduction)

	#Load original movie and 
	enlarge = 50	#Make movie FOV larger than original cropping rectangle by 50pixels or so; otherwies difficult to see what's going on; 
	movie_array = np.load(self.movie_filename, mmap_mode='c')[:, max(0,self.x1-enlarge):self.x2+enlarge, max(0,self.y1-enlarge):self.y2+enlarge]

	#Delete last class
	for k in [3]:
	    temp_indexes = np.where(class_id==k)[0]
	    indexes = np.delete(indexes,temp_indexes,0)
	    class_id = np.delete(class_id,temp_indexes,0)
	
	print indexes
	print class_id
	
	#Load original data
	#Select training data and test data
	#filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+".npy"
	#data_movie = np.load(filename)
	#data_movie = np.float32(self.decimated_frames)
	data_movie = np.float32(self.data)
	print data_movie.shape


	#Convert data to 1D
	X=data_movie[indexes]
	X_1D = []
	for frame in X:
	      X_1D.append(np.ravel(frame))
	X_1D=np.array(X_1D)
	print X_1D.shape 
	X=X_1D
	y = class_id

	test_flag = True
	if test_flag:
	    X_test = data_movie#[:2000]
	    print X_test.shape
	    X_1D = []
	    for frame in X_test:
		  X_1D.append(np.ravel(frame))
	    X_1D=np.array(X_1D)
	    print X_1D.shape 
	    X_test=X_1D
	
	
	C = 1.0
	
	methods = []	
	titles=[]
	
	#Run SVM
	if True: 
	    # SVC with linear kernel
	    print "...computing svm..."
	    classifier = svm.SVC(kernel='linear', C=C, probability=True)
	    svc = classifier.fit(X, y)
	    #svc_test= classifier.predict_proba(X_test)
	    #print svc_test
	    titles.append('SVC with linear kernel')
	    methods.append(svc)

	if False:
	    # LinearSVC (linear kernel)
	    print "...computing svm..."
	    classifier = svm.LinearSVC(C=C)
	    lin_svc = classifier.fit(X, y)
	    #svc_test= classifier.decision_function(X_test)
	    #print svc_test
	    titles.append('LinearSVC (linear kernel)')
	    methods.append(lin_svc)


	if True:
	    # LinearSVC (linear kernel)
	    print "...computing svm..."
	    from sklearn.gaussian_process import GaussianProcessClassifier
	    from sklearn.gaussian_process.kernels import RBF

	    classifier = GaussianProcessClassifier(1.0 * RBF(1.0))
	    gaussian = classifier.fit(X, y)
	    #svc_test= classifier.decision_function(X_test)
	    #print svc_test
	    titles.append('Gaussian Process')
	    methods.append(gaussian)



	if False:
	    # SVC with RBF kernel
	    print "...computing svm..."
	    classifier = svm.SVC(kernel='rbf', gamma=0.7, C=C)
	    rbf_svc = classifier.fit(X, y)
	    titles.append('SVC with RBF kernel')
	    methods.append(rbf_svc)
	
	if True:
	    # SVC with polynomial (degree 3) kernel
	    print "...computing svm..."
	    poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
	    titles.append('SVC with polynomial (degree 3) kernel')
	    methods.append(poly_svc)
		  
	
	colors=['red','blue','green','lightsalmon','dodgerblue','indianred','mediumvioletred','pink', 'brown', 'magenta']
	y_clr = []
	for k in range(len(y)): y_clr.append(colors[y[k]])
	for i, clf in enumerate(methods):
	    # Plot the decision boundary. For that, we will assign a color to each
	    # point in the mesh [x_min, x_max]x[y_min, y_max].
	    plt.subplot(2, 2, i + 1)
	    plt.subplots_adjust(wspace=0.4, hspace=0.4)
	 
	    #plt.scatter(X_test[:,0],X_test[:,1],c=Z, cmap=plt.cm.coolwarm)
	    
	    ## Put the result into a color plot
	    if test_flag:
		Z = clf.predict(X_test)
		Z_unique = np.unique(Z)
		for p in Z_unique:
		    indexes = np.where(Z==Z_unique[p])[0]
		    imageio.mimwrite(self.filename[:-4]+'_'+self.area+'_'+titles[i]+"_cluster"+str(p)+".mp4", movie_array[indexes], fps = self.frame_rate)

		Z_clr = []
		for k in range(len(Z)): Z_clr.append(colors[Z[k]])
		plt.scatter(X_test[:,0], X_test[:,1], c=Z_clr, alpha=0.3)#, cmap=plt.cm.coolwarm, alpha=0.8)
	 
	    # Plot also the training points
	    plt.scatter(X[:,0], X[:,1], c=y_clr)#, cmap=plt.cm.coolwarm)
	    plt.xlabel('length')
	    plt.ylabel('width')

	    plt.xticks(())
	    plt.yticks(())
	    plt.title(titles[i])
	 
	plt.show(block=True)

	
	
    def load_cluster(self, cluster):
	
	print "...loading cluster indexes: ", cluster	
	
	filename = self.filename[:-4]+'_'+self.area+'_'+str(cluster)+".txt"
	self.indexes = np.loadtxt(filename,dtype=np.int32)
	self.data_subsampled = self.data_subsampled[self.indexes]
	
	print self.data_subsampled.shape
	
	
    def subsample_frames(self):
        ''' Further reduce dimensionality by subsampling frames
        '''

        print "...subsampling frames..."
        subsample_filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+"_subsampled.npy"
	decimated_filename = self.filename[:-4]+"_"+self.area+"_cropped_registered_"+self.mode+".npy"
	self.decimated_frames = np.load(decimated_filename)
	
        if os.path.exists(subsample_filename)==False:
            subsampled_array = []
            print "... subsampling ..."    
	    #print self.decimated_frames
            for k in range(len(self.decimated_frames)):
		#print self.decimated_frames[k].shape
                #subsampled_array.append(scipy.misc.imresize(data_2D_array[k], 0.2, interp='bilinear', mode=None))
                subsampled_array.append(scipy.misc.imresize(self.decimated_frames[k], 0.1, interp='bilinear', mode=None))

            self.data_subsampled = np.array(subsampled_array)
            np.save(subsample_filename, self.data_subsampled)

        else:
            self.data_subsampled = np.load(subsample_filename)

    def scale_moving_frames(self):
	
	#Set first value in diff_array to second value (otherwise very large)
	print len(self.diff_array)
	#self.diff_array[0:10]=self.diff_array[10]
	self.diff_array = np.insert(self.diff_array,0,self.diff_array[0])
	self.diff_array-=np.min(self.diff_array)				#Translate to min val=0
	
	print self.data_subsampled.T.shape
	#Scale the 
	self.data_subsampled = (self.data_subsampled.T * self.diff_array).T
	
	print self.data_subsampled.shape

    def load_clusters(self):
	
	self.cluster_names = np.loadtxt(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusternames.txt" ,dtype='str')
	self.cluster_indexes = np.load(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusterindexes.npy")

    def export_clusters(self, recluster_flag=False):
        
	if recluster_flag==True:
	    text_append="_new"
	else:
	    text_append=''
	
	##THIS NEEDS TO BE SIMPLIFIED
	##Subsample frames to further reduce dimensionality and speed up processing
        #if True: self.subsample_frames()
	#else: self.data_subsampled = self.decimated_frames

	##Scale the frame information by some coefficient of movement
	#if False: self.scale_moving_frames()	

        ##Run dim reduction
        #self.dimension_reduction()
        
        ##Filter transformed distributions to remove camera drift (usually)
        #if True:
            #self.filter_PCA(self.data_dim_reduction, filtering=True, plotting=False)
	#self.load_clusters()
	
        ##Legacy variables need to be assigned to object attributes
        ##clustered_pts = self.clustered_pts
        #n_frames= len(self.data_dim_reduction)
        #X = self.data_dim_reduction
        #pathname = self.filename[:-4]
        #area = self.area
        #print "... n frames: ", n_frames
		
        ##Initialize the color_array to all black
        #colors=['red','blue','green','lightsalmon','dodgerblue','indianred','mediumvioletred','pink', 'brown', 'magenta']
        #color_array=[]
        
        ##Initialize a list of clrs;    		NB: crappy way to initialize list of colours; REDO
        #for k in range(n_frames): 
            #color_array.append('mediumvioletred')
        #color_array = np.array(color_array, dtype=str)
        
        ##Enumerate all indexes from 0.. # frames
	#all_frames = np.arange(n_frames)
	
	#try:
	    #self.indexes
	#except:
	    #self.indexes=np.arange(n_frames)	
        
	#List of behaviors:
	cumulative_indexes=[]
	unique_names = np.unique(self.cluster_names)
	print self.cluster_names
	print unique_names
	
	unique_indexes = []
	for ctr1, unique_name in enumerate(unique_names):
	    unique_indexes.append([])
	    for ctr, cluster_name in enumerate(self.cluster_names):
		if unique_name==cluster_name:
		    unique_indexes[ctr1].extend(self.cluster_indexes[ctr])
	
	cumulative_indexes = unique_indexes
	
	#Review the clustering in original PCA space - no point if PCA was done recursively
        #if False:
            #plt.scatter(X[:,0],X[:,1],color=color_array)
            #plt.show(block=True)
	
	#************************************************************************************
	#******************************** GENERATE MOVIES & FIGS ****************************
	#************************************************************************************
        #Load movie 
        enlarge = 50	#Make movie FOV larger than original cropping rectangle by 50pixels or so; otherwies difficult to see what's going on; 
	print self.x1, self.x2, self.y1, self.y2
	print max(0,self.x1-enlarge), self.x2+enlarge, max(0,self.y1-enlarge),self.y2+enlarge
        movie_array = np.load(self.movie_filename, mmap_mode='c')[:, max(0,self.x1-enlarge):self.x2+enlarge, max(0,self.y1-enlarge):self.y2+enlarge]
        
	#movie_array = movie_array[self.indexes]	#Load all or just part of moview --- NOTE IMPLEMNETED
	print movie_array.shape
        #Compute membership in each cluster and save examples to file:
        cluster_ids = []
        dim=4
        frame_indexes = np.arange(len(movie_array))     #Make original indexes and remove them as they are removed from the datasets 
        for k in range(len(cumulative_indexes)):
	    if len(cumulative_indexes[k])==0:
		print "... empty cluster..."
		continue
            img_indexes = np.int32(np.random.choice(cumulative_indexes[k], min(len(cumulative_indexes[k]), dim*dim)))   #Chose random examples from cluster
            
            #Plot individual frames
            gs = gridspec.GridSpec(dim,dim)
            for d in range(len(img_indexes)): 
                ax = plt.subplot(gs[int(d/dim), d%dim])
                plt.imshow(movie_array[img_indexes[d]])#, cmap='Greys_r')
                ax.set_xticks([]); ax.set_yticks([])
            
            plt.suptitle("Cluster: " + unique_names[k] + "/" + str(len(cumulative_indexes))+"  # frames: "+str(len(cumulative_indexes[k])), fontsize = 10)
            plt.savefig(self.filename[:-4]+'_'+self.area+'_cluster_'+unique_names[k]+text_append+'.png')   # save the figure to file
            plt.close() 

            #cluster_name = k
	    np.savetxt(self.filename[:-4]+'_'+self.area+'_cluster_'+unique_names[k]+text_append+".txt", cumulative_indexes[k], fmt='%i')

            print "... making movie: ", unique_names[k], "  # frames: ", len(cumulative_indexes[k])
            imageio.mimwrite(self.filename[:-4]+'_'+self.area+'_cluster_'+unique_names[k]+text_append+".mp4", movie_array[cumulative_indexes[k]], fps = self.frame_rate)
        

    #************************************************************************************************************************************
    #*************************************************UTILTY FUNCTIONS ***************************************************************
    #************************************************************************************************************************************
    #@jit
    def decimate(self, frame, frame_count, rLow, rHigh):
	''' Binarize, erode, dilate and blur each frame
	    TODO: check if alternative approaches work better: e.g. keeping more data in frame
	''' 
        #print frame.shape

        #lower = np.array([0, 0, rLow], dtype = "uint8")
        #upper = np.array([255, 255, rHigh], dtype = "uint8")     
        #lower = np.array([0, rLow, 0], dtype = "uint8")
        #upper = np.array([255, rHigh, 255], dtype = "uint8")     
        lower = np.array([rLow], dtype = "uint8")
        upper = np.array([rHigh], dtype = "uint8")     
        # apply a series of erosions and dilations to the mask
        # using an rectangular kernel
        skinMask = cv2.inRange(frame, lower, upper)

        #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        #skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        if self.show_vid:
            cv2.imshow('original', frame)
            cv2.imshow('filtered', skinMask)
            cv2.waitKey(1)

        #return skinMask
        return frame

    def find_contours(self, frame, frame_count, rLow, rHigh):
	
        lower = np.array([100, 100, rLow], dtype = "uint8")
        upper = np.array([255, 255, rHigh], dtype = "uint8")     
        frame = cv2.inRange(frame, lower, upper)

	#blurred = cv2.pyrMeanShiftFiltering(frame, 31,91)
	#gray = cv2.cvtColor(blurred,cv2.COLOR_BGR2GRAY)
	#ret, threshold = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	ret, threshold = cv2.threshold(frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	
	contours,_=cv2.findContours(threshold,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
	print "... # contours detected: ", len(contours)
	
	cv2.drawContours(frame,contours,0,(0,0,255),-1)
	
	cv2.namedWindow('Display',cv2.WINDOW_NORMAL)
	cv2.imshow('Display',frame)
	cv2.waitKey(1)
	
	#imgray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);
	#ret,thresh = cv2.threshold(imgray,127,255,0);
	#contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);

	##draw a three pixel wide outline 
	#cv2.drawContours(img,contours,-1,(0,255,0),3);
    
    
    def find_edges(self, frame, frame_count, rLow, rHigh):
	
	image = frame
	skinMask = image
	#image = cv2.imread(imagePath)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 3)
	
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	#blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	blurred = skinMask
	image=skinMask
    
	# apply Canny edge detection using a wide threshold, tight
	# threshold, and automatically determined threshold
	wide = cv2.Canny(blurred, 10, 200)
	tight = cv2.Canny(blurred, 225, 250)
	#auto = auto_canny(blurred)
	sigma=0.33
	if True:
	    v = np.median(blurred)
	    # apply automatic Canny edge detection using the computed median
	    lower = int(max(0, (1.0 - sigma) * v))
	    upper = int(min(255, (1.0 + sigma) * v))
	    auto = cv2.Canny(image, lower, upper)
	
        auto = cv2.dilate(auto, kernel, iterations = 2)
	#auto = cv2.erode(auto, kernel, iterations = 3)
	
	# show the images
	#cv2.imshow("Original", image)
	#cv2.imshow("Edges", np.hstack([wide, tight, auto]))
	cv2.imshow("Edges", auto)
	cv2.waitKey(1)
	
	#ax=plt.subplot(1,2,1)
	#ax.imshow(auto)
	#print np.min(auto), np.max(auto)
	#auto=np.nan_to_num(auto)
	#ax=plt.subplot(1,2,2)
	#ax.imshow(auto)
	#print np.min(auto), np.max(auto)
	#print auto
	#plt.show(block=True)
	
	return auto

    #@jit
    def compute_diff(self):
	
	#filename = self.filename[:-4]+"_"+self.area+"_"+"cropped_movie.npy"
	#filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+".npy"
	filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+".npy"
        data = np.load(filename, mmap_mode='c')
        
        self.diff_array = []
        for k in range(len(data)-1):
            print "... computing difference for frame: ", k
            self.diff_array.append(LA.norm(data[k+1]-data[k]))
        
	#Filter video motion to remove any artifact jitters.
        if False: 
	    #filter_val = self.frame_rate*0.49 #Hz
	    filter_val = 0.2	#Fixed threshold filter #Hz
	    self.diff_array = self.butter_lowpass_filter(self.diff_array, filter_val, self.frame_rate, 5)
        
	np.save(self.filename[:-4]+"_diff_array", self.diff_array)

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
        
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def crop_frame_box(self, image, motion_correct_flag = False):
        ''' Function to crop field-of-view of video
        '''
        #global coords, image_temp, ax, fig, cid, img_height, img_width
        
        self.image_temp = image.copy()
        self.img_height, self.img_width = self.image_temp.shape[:2]

	
	if motion_correct_flag:
	    crop_filename = self.filename[:-4]+'_registry_cropped.npz'
        else:
	    crop_filename = self.filename[:-4]+'_'+self.area+'_cropped.npz'
        
	if (os.path.exists(crop_filename)==False):
	    self.fig, self.ax = plt.subplots()
            self.coords=[]

            self.ax.imshow(image)#, vmin=0.0, vmax=0.02)
            if motion_correct_flag:
		self.ax.set_title("Define area to be used for motion registry\n (Click top left + bottom right corner of FOV)")
	    else:
		self.ax.set_title("Define area to be used for image analysis\n (Click top left + bottom right corner of FOV)")
            
	    #figManager = plt.get_current_fig_manager()
            #figManager.window.showMaximized()
            self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
            plt.show(block=True)

            x_coords = np.int32(np.sort([self.coords[0][0], self.coords[1][0]]))
            y_coords = np.int32(np.sort([self.coords[0][1], self.coords[1][1]]))

            #return x_coords[0],x_coords[1],y_coords[0],y_coords[1]
            
	    #if motion_correct_flag==False:
	    self.x1=x_coords[0];  self.x2=x_coords[1];  self.y1=y_coords[0];  self.y2=y_coords[1]
	    np.savez(crop_filename, x1=x_coords[0], x2=x_coords[1], y1=y_coords[0], y2=y_coords[1])	#Save both FOV and registry cropped corners

            print x_coords, y_coords
            if False: 
		self.fig, self.ax = plt.subplots()
                self.ax.imshow(image[self.x1:self.x2, self.y1:self.y2])
                plt.title(self.area +"Field-of-view selected\n(close figure to start)")
                plt.show(block=True)
                
    
    def motion_correct_caiman(self):
	''' Imported rigid motion correction toolbox from caiman
	'''
	fname_registry = self.filename[:-4]+"_registry_cropped.npy"  #This is the cropped FOV specifically for registration
	fname_original = self.filename[:-4]+".npy"				#This is the entire movie
	fname_cropped = self.filename[:-4]+"_"+self.area+"_cropped.npy"	#This is the area FOV 
	#original_mov = np.load(fname_original,mmap_mode='c')
	all_mov = np.load(fname_registry)
	crop_mov = np.load(fname_cropped)
	
	#Check to see if shift_rig file has already been saved
	if os.path.exists(fname_registry[:-4]+"_shifts_rig.npy")==False:

	    # motion correction parameters
	    niter_rig = 1               # number of iterations for rigid motion correction
	    max_shifts = (6, 6)         # maximum allow rigid shift
	    splits_rig = 56             # for parallelization split the movies in  num_splits chuncks across time
	    strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
	    overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
	    splits_els = 30             # for parallelization split the movies in  num_splits chuncks across time
	    upsample_factor_grid = 4    # upsample factor to avoid smearing when merging patches
	    max_deviation_rigid = 3     # maximum deviation allowed for patch with respect to rigid shifts
	    
	    
	    #%% start a cluster for parallel processing
	    #caiman_path = np.loadtxt('caiman_folder_location.txt', dtype=str)       #<------------ is this necessary still?
	    caiman_path = '/home/cat/code/CaImAn'      #<------------ is this necessary still?
	    sys.path.append(str(caiman_path)+'/')
	    print (caiman_path)

	    import caiman as cm
	    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

	    #Load cropped image and use it to align larger movie
	    #original_mov = np.load(fname_original,mmap_mode='c')
	    all_mov = np.load(fname_registry)
	    crop_mov = np.load(fname_cropped)
	    print "...cropped movie shape...", all_mov.shape

	    #return
	    min_mov = all_mov.min()
	    
	    # this will be subtracted from the movie to make it non-negative 

	    from caiman.motion_correction import MotionCorrect
	    mc = MotionCorrect(fname_registry, min_mov,
			   dview=dview, max_shifts=max_shifts, niter_rig=niter_rig,
			   splits_rig=splits_rig, 
			   strides= strides, overlaps= overlaps, splits_els=splits_els,
			   upsample_factor_grid=upsample_factor_grid,
			   max_deviation_rigid=max_deviation_rigid, 
			   shifts_opencv = True, nonneg_movie = True)

	    mc.motion_correct_rigid(save_movie=False,template = None)
	    
	    dview.close()
	    dview.terminate()
	    
	    #matplotlib.use('Agg')	
	    np.save(fname_registry[:-4]+"_shifts_rig.npy", mc.shifts_rig)
	    
	    #Save registered original movie and the registry FOV 
	    reg_mov = np.zeros(all_mov.shape, dtype=np.uint8)
	    #reg_original_mov = np.zeros(original_mov.shape, dtype=np.uint8)
	    for k in range(len(all_mov)):
		if k%1000==0: print "... shifting frame: ",k 
		reg_mov[k] = np.roll(np.roll(all_mov[k], int(mc.shifts_rig[k][0]), axis=0), int(mc.shifts_rig[k][1]), axis=1)
		#reg_original_mov[k] = np.roll(np.roll(original_mov[k], int(mc.shifts_rig[k][0]), axis=0), int(mc.shifts_rig[k][1]), axis=1)

	    #tiff.imsave(fname[:-4]+"_registered.tif", reg_mov)
	    #np.save(fname_original[:-4]+"_registered.npy", reg_original_mov)
	    np.save(fname_registry[:-4]+"_registered.npy", reg_mov)
	
	
	if os.path.exists(fname_cropped[:-4]+"_registered.npy")==False:
	    shifts_rig = np.load(fname_registry[:-4]+"_shifts_rig.npy")

	    print ("... shifting image stack based on motion correction...")

	    reg_cropped_mov = np.zeros(crop_mov.shape, dtype=np.uint8)
	    for k in range(len(all_mov)):
		if k%1000==0: print "... shifting frame: ",k 
		reg_cropped_mov[k] = np.roll(np.roll(crop_mov[k], int(shifts_rig[k][0]), axis=0), int(shifts_rig[k][1]), axis=1)
		
	    np.save(fname_cropped[:-4]+"_registered.npy", reg_cropped_mov)

	    imageio.mimwrite(fname_registry[:-4]+"_registered.mp4", reg_cropped_mov, fps = self.frame_rate)


	if False:
	    print "...saving movies..."
	    imageio.mimwrite(fname_original[:-4]+"_registered.mp4", reg_original_mov, fps = self.frame_rate)
	    imageio.mimwrite(fname_registry[:-4]+"_registered.mp4", reg_mov, fps = self.frame_rate)
	    imageio.mimwrite(fname_cropped[:-4]+"_registered.mp4", reg_cropped_mov, fps = self.frame_rate)


    def register(self):
	import imreg_dft as ird

	enlarge=0
	#data = np.load(filename[:-4]+"_movie.npy",mmap_mode='c')[:,:,:,1]

	#Convert original file to .npy first
	if True:  #Use binarized 
	    #self.movie_filename = self.filename[:-4]+"_"+self.area+"_"+self.mode+".npy" #This is cropped + binarized image
	    self.movie_filename = self.filename[:-4]+'_'+self.area+'_cropped_movie.npy'	#This is cropped original image
	    data = np.load(self.movie_filename, mmap_mode='c')[:,:,:,1] #pick green channel
	else:
	    self.movie_filename = self.filename[:-4]+'.npy'	#Check if original video file was converted to .npy format
	    data = np.load(self.movie_filename, mmap_mode='c')[:,:,:,1]
	
	original_data = np.load(self.filename[:-4]+'.npy', mmap_mode='c')[:,:,:,1]
	#Register the original data
	#Note: using either cropped image or entire video frame need different step
	registered_filename = self.filename[:-4]+"_registered.npy"
	if os.path.exists(registered_filename)==False:

	    #data = self.movie[:,:,:,1]
	    im0=data[0]
	    result_array=[]
	    result_array.append(original_data[0])
	    #for k in range(len(data[:100])):
	    for k in range(len(data)):
		if k%100==0: print "...registering frame: ", k
		im1= data[k]
		tvec = ird.translation(im0,im1)["tvec"].round(4)
		result_array.append(np.uint8(ird.transform_img(original_data[k], tvec=tvec)))	#Perform transformation on entire recording

	    #print result_array
	    #result_array=np.uint8(result_array)
	    #print result_array.shape
	    print "...saving registered data ..."
	    np.save(registered_filename, result_array)
	    imageio.mimwrite(registered_filename[:-4]+'.mp4', result_array, fps = 10.)
	
    def on_click(self, event):
        ''' Mouse click function that catches clicks and plots them on top of existing image
        '''
        #global coords, image_temp, ax, fig, cid
        
        print self.coords
        if event.inaxes is not None:
            print event.ydata, event.xdata
            self.coords.append((event.ydata, event.xdata))
            for j in range(len(self.coords)):
                for k in range(6):
                    for l in range(6):
                        #print self.coords[j][0], self.coords[j][1]
                        #print  image_temp[int(event.ydata)-1+k,int(event.xdata)-1+l]
                        self.image_temp[int(event.ydata)-1+k,int(event.xdata)-1+l]=np.max(self.image_temp)

            self.ax.imshow(self.image_temp)
            self.fig.canvas.draw()
        else:
            print 'Exiting'
            plt.close()
            self.fig.canvas.mpl_disconnect(self.cid)


    def on_click_classify(self, event):
        ''' Mouse click function to click on four corners 
        '''        

	#Clicks inside the image
        if event.inaxes is not None:
	    print "Clicked inside"
            print event.ydata, event.xdata
	    self.coords = [event.ydata, event.xdata]

            plt.close()
            fig.canvas.mpl_disconnect(self.cid)

	    
	#Clicks outside image
        else:
            print 'Clicked Outside'
            plt.close()
            fig.canvas.mpl_disconnect(self.cid)
	    self.complete = True
	    
    
    def dimension_reduction(self):
        ''' Running dimensionality reduction on the 
            A list of frame indexes can be provided, otherwise all frames will be considered;
        ''' 
        
        print "... computing original dim reduction ..."

	#self.subsample_frames()

	frames_list = np.arange(len(self.data_subsampled))

        #Convert data to 1D vectors before dim reduction
        self.data_subsampled_1D= []
        for k in frames_list:
            self.data_subsampled_1D.append(np.ravel(self.data_subsampled[k]))
        
        filename = self.filename[:-4]+'_'
        area = self.area			#Correct
        method=self.methods[self.method]
        matrix_in = self.data_subsampled_1D
        
        print "Computing dim reduction, size of array: ", np.array(matrix_in).shape
        
        if self.method==0:
            #MDS Method - SMACOF implementation Nelle Varoquaux
            if os.path.exists(filename+area+'_'+method+'.npy')==False:
            
                print "... MDS-SMACOF..."
                print "... pairwise dist ..."
                dists = metrics.pairwise.pairwise_distances(matrix_in)
                adist = np.array(dists)
                amax = np.amax(adist)
                adist /= amax
                
                print "... computing MDS ..."
                mds_clf = manifold.MDS(n_components=2, metric=True, n_jobs=-1, dissimilarity="precomputed", random_state=6)
                results = mds_clf.fit(adist)
                Y = results.embedding_         
                 
                np.save(filename+area+'_'+method, Y)
            
            else:
                Y = np.load(filename+area+'_'+method+'.npy')


        elif self.method==1:
            ##t-Distributed Stochastic Neighbor Embedding; Laurens van der Maaten
            if os.path.exists(filename+area+'_'+method+'.npy')==False:

                print "... tSNE ..."
                print "... pairwise dist ..."
                
                dists = sklearn.metrics.pairwise.pairwise_distances(matrix_in)
                
                adist = np.array(dists)
                amax = np.amax(adist)
                adist /= amax
                
                print "... computing tSNE ..."
                model = manifold.TSNE(n_components=3, init='pca', random_state=0)
                Y = model.fit_transform(adist)
                
                np.save(filename+area+'_'+method, Y)
            
            else:
                Y = np.load(filename+area+'_'+method+'.npy')

        elif self.method==2:

            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                Y = self.PCA_reduction(matrix_in, 3)
                np.save(filename+area+'_'+method, Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')
                
        elif self.method==3:

            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                print "... computing Barnes-Hut tSNE..."
                Y = bh_sne(np.float64(matrix_in), perplexity=90.)
                
                np.save(filename+area+'_'+method+'.npy', Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')

        
        elif self.method==4:
            
            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                print "... computing locally linear embedding ..."
                n_neighbors = 30
                
                clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='standard')
                Y = clf.fit_transform(np.float64(matrix_in))
                
                np.save(filename+area+'_'+method+'.npy', Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')

        elif self.method==5:
            
            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                print "... computing Hessian locally linear embedding ..."
                n_neighbors = 30
                
                clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='hessian')
                Y = clf.fit_transform(np.float64(matrix_in))
                
                np.save(filename+area+'_'+method+'.npy', Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')

        elif self.method==6:
            
            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                print "... computing Hessian locally linear embedding ..."
                n_neighbors = 30
                
                clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
                              method='ltsa')
                Y = clf.fit_transform(np.float64(matrix_in))
                
                np.save(filename+area+'_'+method+'.npy', Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')

        elif self.method==7:
            
            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                print "... computing Random Trees embedding locally linear embedding ..."

                hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0,
                                       max_depth=5)
                X_transformed = hasher.fit_transform(np.float64(matrix_in))
                pca = decomposition.TruncatedSVD(n_components=2)
                Y = pca.fit_transform(X_transformed)

                np.save(filename+area+'_'+method+'.npy', Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')


        elif self.method==8:
            
            if os.path.exists(filename+area+'_'+method+'.npy')==False:
                print "... computing Spectral embedding ..."

                embedder = manifold.SpectralEmbedding(n_components=2, random_state=0,
                                      eigen_solver="arpack")
                Y = embedder.fit_transform(np.float64(matrix_in))


                np.save(filename+area+'_'+method+'.npy', Y)
            else:
                Y = np.load(filename+area+'_'+method+'.npy')

        self.data_dim_reduction = Y


    def cluster_data(self, indexes='all'): #cluster_data, cluster_method, dim_reduction_method):
        
        #colours = ['blue','red','green','black','orange','magenta','cyan','yellow','brown','pink','blue','red','green','black','orange','magenta','cyan','yellow','brown','pink','blue','red','green','black','orange','magenta','cyan','yellow','brown','pink']
        
        #cluster_method=2
        
        #cluster_method = self.cluster_method
        #if indexes=='all':
	#    data = self.data_dim_reduction
        #else:
	#    data = self.data_dim_reduction[indexes]	#Cluster only part of data
	    
	#MANUAL
        if self.cluster_method == 3: 
            self.clustered_pts = self.manual_cluster(indexes=indexes)		#not sure the self attribute is still required
            return 
	else:
	    print "... clustering method not implemented ..."
	    quit()
	   
        #labels = np.array(labels)
        #clrs = []
        #for k in range(len(labels)):
            #clrs.append(colours[labels[k]])
        #plt.scatter(data[:,0], data[:,1], color=clrs)
        #plt.show(block=True) 
	    
	#Automated clustering not working
	#KMEANS
        #if cluster_method == 0: 
            #from sklearn import cluster

            #n_clusters = 3
            #print "... n_clusters sought: ", 3
            #clusters = cluster.KMeans(n_clusters, max_iter=1000, n_jobs=-1, random_state=1032)
            #clusters.fit(data)

            #labels = clusters.labels_
            
            #clustered_pts = []
            ##for k in range(len(np.max(labels)):
            
            #print " TODO : FIX THIS TO CORRESPOND TO RECURSIVE CLUSTERING ..."
            #return labels, labels
        
        ##MEAN SHIFT
        #if cluster_method == 1: 
            #from sklearn.cluster import MeanShift, estimate_bandwidth
            #from sklearn.datasets.samples_generator import make_blobs
            
            #quantile = 0.1
            #bandwidth = estimate_bandwidth(data, quantile=quantile, n_samples=5000)

            #ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            #ms.fit(data)
            #labels = ms.labels_
            ##print labels

        ##DBSCAN
        #if cluster_method == 2: 
            #from sklearn.cluster import DBSCAN
            #from sklearn import metrics
            #from sklearn.datasets.samples_generator import make_blobs
            #from sklearn.preprocessing import StandardScaler 

            #X = StandardScaler().fit_transform(data)

            #eps = 0.2
            
            #db = DBSCAN(eps=eps, min_samples=10).fit(X)
            #core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            #core_samples_mask[db.core_sample_indices_] = True
            #labels = db.labels_ 


            


    def filter_PCA(self, X, filtering, plotting): 
	print " ... filtering PCA data ..."
        #print " ... xshape:" , X.shape

        #Filter PCA DATA
	plt.close()
        for d in range(X.shape[1]):
            if plotting: 
                t = np.linspace(0,len(X[:,0]), len(X[:,0]))/self.frame_rate
                ax = plt.subplot(X.shape[1],1,d+1)

                plt.plot(t, X[:,d])
                plt.title("PCA #"+str(d)+" vs. time (sec)", fontsize = 20)

            if filtering: 
                x = np.hstack(X[:,d])
                b, a = butter(2, 0.001, 'high') #Implement 2Hz highpass filter
                y = filtfilt(b, a, x)
                X[:,d]=y
            
                if plotting: 
                    plt.plot(t, y, color='red')
            
        if plotting: 
            plt.show(block=True)
        
        self.data_dim_reduction = X.copy()	#Need to also copy this data into the separate attribute because filtering is optional 

    def filter_data(self, X, filtering=True, plotting=True):
	
	print " ... xshape:" , X.shape

        #Convert data to 1D vectors before dim reduction
        self.data_subsampled_1D= []
        for k in range(len(X)):
            self.data_subsampled_1D.append(np.ravel(X[k]))
	
	X = np.array(self.data_subsampled_1D)
	    
        #Filter PCA DATA
        for d in range(X.shape[1]):
            if plotting: 
                t = np.linspace(0,len(X[:,0]), len(X[:,0]))/self.frame_rate
                ax = plt.subplot(X.shape[1],1,d+1)

                plt.plot(t, X[:,d])
                plt.title("PCA #"+str(d)+" vs. time (sec)", fontsize = 20)

            if filtering: 
                x = np.hstack(X[:,d])
                b, a = butter(2, 0.001, 'high') #Implement 2Hz highpass filter
                y = filtfilt(b, a, x)
                X[:,d]=y
            
                if plotting: 
                    plt.plot(t, y, color='red')
            
        if plotting: 
            plt.show(block=True)
        
        self.data_dim_reduction = X.copy()	#Need to also copy this data into the separate attribute because filtering is optional 



    def manual_cluster(self, indexes='all'):
        
	#Load video data to display during clustering
	enlarge=50
	try: 
	    self.display_frames = np.load(self.filename[:-4]+'_registered.npy')[:, max(0,self.x1-enlarge):self.x2+enlarge, max(0,self.y1-enlarge):self.y2+enlarge]
	except:
	    self.display_frames = np.load(self.filename[:-4]+'.npy',mmap_mode='c')[:, max(0,self.x1-enlarge):self.x2+enlarge, max(0,self.y1-enlarge):self.y2+enlarge]
	    print "... missing registered file, loading original..."

	#Select all indexes; or just subset for re-clustering
	if indexes=='all':
	    self.data = self.data_dim_reduction
	else:
	    self.data=self.data_dim_reduction[indexes]
	    self.display_frames = self.display_frames[indexes]
	
	ctr = 0			#Keep track of cluster #
        clustered_pts = []  	#keep track of clsutered points - NOTE THIS IS RECURSIVE: i.e., the points have relative indexes not absolute

	#need to keep track of indexes as they are being deleted
	self.frame_indexes = np.arange(len(self.data))
	self.cluster_names = []
	self.cluster_indexes = []
	while True:
	    cmap_array = np.linspace(len(self.data)/4,len(self.data),len(self.data))/float(len(self.data))#*256.
            clustered_pts.append([])
            print "... NEW LOOP..."
            plt.close('all')

	    self.fig = plt.figure()
	    self.ax1 = self.fig.add_subplot(121) #plt.subplot(1,2,1)
	    self.img = self.ax1.imshow(self.display_frames[0],vmin=0, vmax=255, animated=True)
	    
	    self.ax2 = self.fig.add_subplot(122) #plt.subplot(1,2,2)
	    self.coords=[]
	    
	    #mng = plt.get_current_fig_manager()
	    #mng.resize(*mng.window.maxsize())
	
	    #self.ax2.scatter(data[:,0],data[:,1], color='black', picker=True)
            self.ax2.scatter(self.data[:,0],self.data[:,1], color = cm.viridis(cmap_array), alpha=0.5, picker=True)
	    
            self.ax2.set_title("Manual select clusters")
            #self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click_single_frame)
            self.cid = self.fig.canvas.mpl_connect('motion_notify_event', self.on_plot_hover)

	    plt.show(block=True)

	    self.cluster_names.append(raw_input("Cluster label: "))
	    
            print "... COORDS OUT: ", self.coords
            print "... data...", self.data
	    
            #if len(self.coords)==0:
            if self.cluster_names[ctr]=='rest':
                print "... DONE MANUAL CLUSTERING ..."

		#Save remaining indexes to file
		self.cluster_indexes.append(self.frame_indexes)
		
		#Reconcile all the duplicate clusters:
		#Assign clusters to unique ids
		cumulative_indexes=[]
		unique_names = np.unique(self.cluster_names)
		print self.cluster_names
		print unique_names
		
		unique_indexes = []
		for ctr1, unique_name in enumerate(unique_names):
		    unique_indexes.append([])
		    for ctr, cluster_name in enumerate(self.cluster_names):
			if unique_name==cluster_name:
			    unique_indexes[ctr1].extend(self.cluster_indexes[ctr])
	    
		#cluster_id = np.where(unique_names==cluster)[0]
		#print "... cluster_id: ", cluster_id
		
		#Save both names and indexes
		if indexes=='all':		#If saving all original data
		    
		    #Reconcile all the duplicate clusters; Assign clusters to unique ids
		    cumulative_indexes=[]
		    unique_names = np.unique(self.cluster_names)
		    print self.cluster_names
		    print unique_names
		    
		    unique_indexes = []
		    for ctr1, unique_name in enumerate(unique_names):
			unique_indexes.append([])
			for ctr, cluster_name in enumerate(self.cluster_names):
			    if unique_name==cluster_name:
				unique_indexes[ctr1].extend(self.cluster_indexes[ctr])
				    
		    np.savetxt(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusternames.txt", unique_names,fmt='%s')
		    np.save(self.filename[:-4]+"_"+self.area+"_"+self.methods[self.method]+"_clusterindexes.npy", unique_indexes)

		else:				#If making edits to original data
		    print '... adjusting previous indexes...'
		    self.resave_clusters(indexes=indexes)

                return clustered_pts		#Return list of lists of points clustered in each stage; 
        
	    if len(self.coords)==0: 
		self.cluster_indexes.append([])
		plt.close()
		ctr+=1
		continue #Skip computation of stuff before and redraw
            
	    
	    #FIND points inside polygon
            bbPath = Path(np.array(self.coords))
            for k,d in enumerate(self.data):
                if bbPath.contains_point(d):
                    clustered_pts[ctr].append(k)
	    
	    if len(clustered_pts[ctr])==0: 
		self.cluster_indexes.append([])
		plt.close()
		ctr+=1
		continue #Skip computation of stuff before and redraw
            
	    #Delete the clustered_pts indexes from 
            print self.data.shape
            self.data = np.delete(self.data, clustered_pts[ctr], axis=0)
	    self.cluster_indexes.append(self.frame_indexes[np.array(clustered_pts[ctr])])
	    self.frame_indexes = np.delete(self.frame_indexes, clustered_pts[ctr], axis=0)
            print self.data.shape

            methods = ['MDS', 'tSNE', 'PCA', 'BHtSNE']
            method = methods[2]
            print "... recomputing dim reduction on remaning scatter plot ..."

            if self.methods[self.method] !='tSNE':
                n_components=2
                self.data = self.PCA_reduction(self.data, n_components)

            plt.close()
            ctr+=1

        print ".... WRONG EXIT ********"

    #def on_plot_scatter_click(self,event):
	##print self.ax2.get_lines()
	#ind = event.ind
        ##print 'onpick3 scatter:', ind, np.take(self.data_dim_reduction[:,0], ind), np.take(self.data_dim_reduction[:,1], ind)
        #print 'scatter:', ind[0]
	
	#self.img.set_data(self.display_frames[ind[0]])
	#plt.draw()
	

    def on_plot_hover(self,event):
	print event.xdata, event.ydata
	
	a = self.data[:,:2]
	index = distance.cdist([(event.xdata,event.ydata)], a).argmin()
	self.img.set_data(self.display_frames[self.frame_indexes[index]])
	plt.draw()

	if event.button==1:
	    print "***********"
	    #print event.xdata, event.ydata
	    self.coords.append([event.xdata, event.ydata])
            self.ax2.scatter(event.xdata, event.ydata, color='red', s=50)
            self.fig.canvas.draw()
	    time.sleep(.1)
	    
        #if (event.inaxes is None) and (event.button==1):
        if event.button==2:
            print 'Exiting'
            plt.close()
            self.fig.canvas.mpl_disconnect(self.cid)
	    time.sleep(.1)

    
    def on_click_single_frame(self,event):
        #global coords, ax, fig, cid, temp_img
        
        if event.inaxes is not None:
	    print event.xdata, event.ydata
            self.coords.append([event.xdata, event.ydata])
            self.ax2.scatter(event.xdata, event.ydata, color='red', s=50)
            self.fig.canvas.draw()

        else:
            print 'Exiting'
            plt.close()
            self.fig.canvas.mpl_disconnect(self.cid)

    
    def PCA_reduction(self, X, n_components):
        ''' Redundant function, can just use dim-redution fuction above
        '''
        
        plt.cla()
        #pca = decomposition.SparsePCA(n_components=3, n_jobs=1)
        pca = decomposition.PCA(n_components=n_components)

        print "... fitting PCA ..."
        pca.fit(X)
        
        return pca.transform(X)
            
def plot_metadata(output):
	
    tot_frames = 925566

    #Convert dta to pretty plots
    output.drift_matrix[::output.scale]=np.nan
    output.drift_matrix[:,::output.scale]=np.nan
    output.spout_matrix[::output.scale]=np.nan
    output.spout_matrix[:,::output.scale]=np.nan
    output.lick_matrix[::output.scale]=np.nan
    output.lick_matrix[:,::output.scale]=np.nan
    output.paw_matrix[::output.scale]=np.nan
    output.paw_matrix[:,::output.scale]=np.nan
    output.scratch_matrix[::output.scale]=np.nan
    output.scratch_matrix[:,::output.scale]=np.nan

    output.lick_matrix*=100
    output.paw_matrix*=100
    output.scratch_matrix*=100

    x_labels = []
    for root_dir in output.root_dirs:
	x_labels.append(os.path.split(root_dir)[1])
    y_labels = []
    for k in range(1,22):
	y_labels.append(k)

    fig = plt.figure()
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    fontsize=20
    if False:
	ax=plt.subplot(1,2,1)
	plt.imshow(output.drift_matrix.T)
	plt.ylim(0,21*output.scale)
	plt.title("Camera Drifts During Recording", fontsize=fontsize)
	plt.ylabel("Session #", fontsize=fontsize)
	plt.xlabel("Animal ID", fontsize=fontsize)
	plt.xticks(np.int16(range(0,len(output.root_dirs)*output.scale,output.scale))+scale/3, x_labels,rotation=70)
	plt.yticks(np.int16(range(0,21*output.scale,output.scale))+output.scale/2, y_labels)
	plt.tick_params(axis='both', which='both', labelsize=fontsize-5)

	ax=plt.subplot(1,2,2)
	plt.imshow(output.spout_matrix.T)
	plt.ylim(0,21*output.scale)
	plt.title("Spout Moves During Recording", fontsize=fontsize)
	plt.ylabel("Session #", fontsize=fontsize)
	plt.xlabel("Animal ID", fontsize=fontsize)
	plt.xticks(np.int16(range(0,len(output.root_dirs)*output.scale,output.scale))+output.scale/3, x_labels,rotation=70)
	plt.yticks(np.int16(range(0,21*output.scale,output.scale))+output.scale/2, y_labels)
	plt.tick_params(axis='both', which='both', labelsize=fontsize-5)
	plt.show()
    else:
	ax=plt.subplot(1,3,1)
	plt.imshow(output.lick_matrix.T)
	plt.ylim(0,21*output.scale)
	plt.title("% Frames Licking", fontsize=fontsize)
	plt.ylabel("Session #", fontsize=fontsize)
	plt.xlabel("Animal ID", fontsize=fontsize)
	plt.xticks(np.int16(range(0,len(output.root_dirs)*output.scale,output.scale))+output.scale/3, x_labels,rotation=70)
	plt.yticks(np.int16(range(0,21*output.scale,output.scale))+output.scale/2, y_labels)
	plt.tick_params(axis='both', which='both', labelsize=fontsize-5)
	cbaxes = inset_axes(ax, width="8%", height="10%") 
	plt.colorbar(cax=cbaxes, ticks=[0.,int(np.nanmax(output.lick_matrix))], orientation='vertical')

	ax=plt.subplot(1,3,2)
	plt.imshow(output.paw_matrix.T)
	plt.ylim(0,21*output.scale)
	plt.title("% Frames Pawing", fontsize=fontsize)
	#plt.ylabel("Session #", fontsize=fontsize)
	plt.xlabel("Animal ID", fontsize=fontsize)
	plt.xticks(np.int16(range(0,len(output.root_dirs)*output.scale,output.scale))+output.scale/3, x_labels,rotation=70)
	plt.yticks(np.int16(range(0,21*output.scale,output.scale))+output.scale/2, y_labels)
	plt.tick_params(axis='both', which='both', labelsize=fontsize-5)
	cbaxes = inset_axes(ax, width="8%", height="10%") 
	plt.colorbar(cax=cbaxes, ticks=[0.,int(np.nanmax(output.paw_matrix))], orientation='vertical')

	ax=plt.subplot(1,3,3)
	plt.imshow(output.scratch_matrix.T)
	plt.ylim(0,21*output.scale)
	plt.title("% Frames Scratching", fontsize=fontsize)
	#plt.ylabel("Session #", fontsize=fontsize)
	plt.xlabel("Animal ID", fontsize=fontsize)
	plt.xticks(np.int16(range(0,len(output.root_dirs)*output.scale,output.scale))+output.scale/3, x_labels,rotation=70)
	plt.yticks(np.int16(range(0,21*output.scale,output.scale))+output.scale/2, y_labels)
	plt.tick_params(axis='both', which='both', labelsize=fontsize-5)
	cbaxes = inset_axes(ax, width="8%", height="10%") 
	plt.colorbar(cax=cbaxes, ticks=[0.,int(np.nanmax(output.scratch_matrix))], orientation='vertical')

	plt.show()


class emptyObject(object):
    pass
    
    
