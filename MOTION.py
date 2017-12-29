import numpy as np
import matplotlib.pyplot as plt
import cv2, os
#from numba import jit
from numpy import linalg as LA
from scipy.signal import butter, filtfilt, cheby1


#**************************************************************************************************************************
#*************************************************CODE START***************************************************************
#**************************************************************************************************************************

filename ='/media/cat/4TB_ephys/in_vivo/james/ANM303491_Ph201/20151029-1425-A1/output.avi'
print filename

class MOTION(object):
    ''' Class to detect motion in behaviour video;
        self.crop() to select only part of video (speeds up analysis)
        self.dimreduce() to reduce dimensionality of video and increase SNR
        self.detect_motion() compute euclidean distance between frames and plots timecourse
    '''
    
    def __init__(self,filename):
        print "...loading: ", filename
        self.filename = filename

    def crop(self):
        #**************** SELECT CROPPED AREA TO TRACK MOTION (smaller is faster) **********************
        #Load sample frame 
        camera = cv2.VideoCapture(self.filename)
        ret,frame = camera.read()
        image_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.destroyAllWindows()
        camera.release()

        #Run cropping functions on sample frame
        self.x1,self.x2,self.y1,self.y2 = self.crop_frame_box(image_original, self.filename)      #DEFINE BOX AREAS FOR CROPPING

    def dimreduce(self):

        #********************* Dim reduction of video using cv2 functions ****************
        if os.path.exists(self.filename[:-4]+"_reduced.npy")==False:
            
            show_vid = False  #Flag to look at output of decimation process

            # define the upper and lower boundaries of the HSV pixel
            # intensities to be considered 'skin'
            rLow = 100 #cv2.getTrackbarPos('R-low', 'images')
            rHigh = 255 #cv2.getTrackbarPos('R-high', 'images')

            camera = cv2.VideoCapture(self.filename)
            reduced_frames = []
            frame_count = 0
            while True:
                print frame_count; frame_count += 1

                (grabbed, frame) = camera.read()
                if not grabbed: break
                
                reduced_frames.append(self.decimate(frame, frame_count, show_vid, rLow, rHigh,self.x1,self.x2,self.y1,self.y2))
                
            np.save(self.filename[:-4]+"_reduced", reduced_frames)
        else:
            diff_array = np.load(self.filename[:-4]+"_diff_array.npy")

    def detect_motion(self):
        #************************ Compute euclidean distance between frames *****************

        if os.path.exists(self.filename[:-4]+"_diff_array.npy")==False:
            diff_array = self.compute_diff(self.filename)
            np.save(self.filename[:-4]+"_diff_array", diff_array)
        else:
            diff_array = np.load(self.filename[:-4]+"_diff_array.npy")

        #Filter video motion to remove any artifact jitters.
        filter_val = 0.2 #Hz
        vid_fps = 30
        diff_array = self.butter_lowpass_filter(diff_array, filter_val, vid_fps, 5)

        #Plot motion as a function of time @ 30Hz
        t = np.arange(len(diff_array))/(30.)
        plt.plot(t, diff_array)

        #Plotting parameters
        plt.xlim(0,t[-1])
        plt.yticks([])
        font_size = 20
        plt.xlabel("Time (sec)", fontsize = font_size)
        plt.ylabel("Movement index (a.u.)", fontsize = font_size)
        plt.tick_params(axis='both', which='both', labelsize=font_size)
        plt.title(self.filename, fontsize = font_size)
        plt.show()

    #************************************************************************************************************************************
    #*************************************************UTILTY FUNCTIONS ***************************************************************
    #************************************************************************************************************************************
    #@jit
    def decimate(self, frame, frame_count, show_vid, rLow, rHigh,x1,x2,y1,y2):
        frame = frame[x1:x2,y1:y2]
        #print frame.shape

        lower = np.array([0, 0, rLow], dtype = "uint8")
        upper = np.array([255, 255, rHigh], dtype = "uint8")     
        # apply a series of erosions and dilations to the mask
        # using an rectangular kernel
        skinMask = cv2.inRange(frame, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        skinMask = cv2.erode(skinMask, kernel, iterations = 1)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 1)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (5, 5), 0)

        if show_vid:
            cv2.imshow('original', frame)
            cv2.imshow('filtered', skinMask)
            cv2.waitKey(1)

        return skinMask

    #@jit
    def compute_diff(self, filename):
        data = np.load(filename[:-4]+"_reduced.npy", mmap_mode='c')
        
        diff_array = []
        for k in range(len(data)-1):
            print k
            diff_array.append(LA.norm(data[k+1]-data[k]))
        
        return diff_array

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        normal_cutoff = cutoff/nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a
        
    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    def crop_frame_box(self, image, filename):

        global coords, image_temp, ax, fig, cid, img_height, img_width
        
        image_temp = image.copy()
        img_height, img_width = image_temp.shape[:2]

        fig, ax = plt.subplots()

        if (os.path.exists(filename[:-4]+'_crop.npy')==False):
            coords=[]

            ax.imshow(image)#, vmin=0.0, vmax=0.02)
            ax.set_title("Click top left corner and bottom right corner to crop image\n (close figure or click off image to end)")
            #figManager = plt.get_current_fig_manager()
            #figManager.window.showMaximized()
            cid = fig.canvas.mpl_connect('button_press_event', self.on_click)
            plt.show()

            fig, ax = plt.subplots()
            x_coords = np.int32(np.sort([coords[0][0], coords[1][0]]))
            y_coords = np.int32(np.sort([coords[0][1], coords[1][1]]))
            print x_coords, y_coords
            ax.imshow(image[x_coords[0]:x_coords[1], y_coords[0]: y_coords[1]])
            plt.title("Cropped field-of-view that will be analyzed\n(close figure to start)")
            
            coords_out = np.vstack((x_coords, y_coords))
            plt.show()
                    
            return x_coords[0],x_coords[1],y_coords[0],y_coords[1]
            
    def on_click(self, event):
        global coords, image_temp, ax, fig, cid
        
        print coords
        if event.inaxes is not None:
            print event.ydata, event.xdata
            coords.append((event.ydata, event.xdata))
            for j in range(len(coords)):
                for k in range(6):
                    for l in range(6):
                        print coords[j][0], coords[j][1]
                        #print  image_temp[int(event.ydata)-1+k,int(event.xdata)-1+l]
                        image_temp[int(event.ydata)-1+k,int(event.xdata)-1+l]=np.max(image_temp)

            ax.imshow(image_temp)
            fig.canvas.draw()
        else:
            print 'Exiting'
            plt.close()
            fig.canvas.mpl_disconnect(cid)
