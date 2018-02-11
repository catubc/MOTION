#Toolbox to detect motion in behaviour video

from MOTION import MOTION
from MOTION import plot_metadata, emptyObject
import glob, os
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')




#**************** Load video data ****************
filename = '/media/cat/250GB/in_vivo/yuki/AI3/videos/AI3_2014-10-28_15-26-09.219/AI3_2014-10-28 _15-26-09.219.wmv'

#****************load meta data ******************
root_dirs = ['/media/cat/250GB/in_vivo/yuki/AI3/videos/'
]

output = emptyObject()
output.scale = 20
output.drift_matrix = np.zeros((len(root_dirs)*output.scale,21*output.scale),dtype=np.int8)*np.nan
output.spout_matrix = np.zeros((len(root_dirs)*output.scale,21*output.scale),dtype=np.int8)*np.nan
output.lick_matrix = np.zeros((len(root_dirs)*output.scale,21*output.scale),dtype=np.int8)*np.nan
output.paw_matrix = np.zeros((len(root_dirs)*output.scale,21*output.scale),dtype=np.int8)*np.nan
output.scratch_matrix = np.zeros((len(root_dirs)*output.scale,21*output.scale),dtype=np.int8)*np.nan
output.root_dirs = root_dirs

img_array = []
titles_array = []
for rt_ctr, root_dir in enumerate(root_dirs):
    sessions = sorted(glob.glob(root_dir+'/*'))
    for ses_ctr,session in enumerate(sessions):
	if 'notes' in session: continue
	session_name = os.path.split(session)[1]
	if session_name[0]=="M":
	    filename = session+'/'+session_name[1:]+'.avi'
	else:
	    filename = session+'/'+session_name+'.avi'
	    #filename = session+'/'+session_name+'.wmv'

	print filename
	#************************************************
	#******************* SET PARAMETERS *************
	#************************************************
	mot = MOTION(filename)		#Asigns mot.filename = filename
	mot.show_vid = True		#Flag to see videos during dim_reduction process
	#mot.frame_rate = 9.375058	#Frame rate, now loading from .avi metadata
	mot.area = 'mouth'		#Name of area to be cropped
	mot.plot_3D = False		#Option to cluster data in 3D using opengl routines; not yet adopted
	mot.mode='all'			#select mode for reducing frames: none, contours, edges
	mot.methods = ['MDS', 'tSNE', 'PCA', 'BHtSNE', 'LLE','HLLE', 'LTSA', 'TRTE','SE']
	mot.method = 2
	mot.rt_ctr = rt_ctr
	mot.ses_ctr = ses_ctr
	#************************************************
	#***************** PROCESS VIDEO ****************
	#************************************************
	if True:
	#if '20160921_013' in filename:
	#if filename == '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_007/M20161205_007.avi':
	    mot.crop()					#Crop video to area indicated above
	    mot.motion_correct_caiman()			#Caiman module for motion correction
	    mot.binarize_frames()				#Reduce the video: none, contours, edges 
	    #mot.detect_movement()				#Detect movement by computing euclidean distance 
	    mot.annotate_frames()				#Annotate movie frames; calls multiple functions	
	    #mot.recluster_id = 'rest'			#Re-annotate some of the frames clustered above
	    #mot.resplit_cluster(cluster=mot.recluster_id)			
	    mot.save_metadata()
	    pass
	    
	#************************************************
	#***************** RELOAD META DATA ************
	#************************************************
	#output = mot.read_metadata(output)
	behaviour = 'mouth_open'
	img = mot.load_frames(behaviour)
	if img != None:
	    img_array.append(img)
	    titles_array.append(session_name)

n_plots = 3
ctr=0
fig = plt.figure()
fig.tight_layout()
indexes = np.random.choice(np.arange(len(img_array)),n_plots*n_plots)
plt.suptitle("Examples of "+behaviour+" frames", fontsize=25)
for k in range(n_plots):
    for p in range(n_plots):
	ax=plt.subplot(n_plots,n_plots,ctr+1)
	plt.title(titles_array[indexes[ctr]])
	plt.imshow(img_array[indexes[ctr]])
	ax.get_xaxis().set_visible(False); ax.yaxis.set_ticks([]); ax.yaxis.labelpad = 0

	ctr+=1

plt.show()
#Plot data
plot_metadata(output)
