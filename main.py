#Toolbox to detect motion in behaviour video

from MOTION import MOTION
from MOTION import plot_metadata, emptyObject
import glob, os
import matplotlib.pyplot as plt
import numpy as np
#import matplotlib
#matplotlib.use('TkAgg')

#************************************************
#********************* OLD LIST ********************
#************************************************
#M20160226
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_003/20160226_003.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_004/20160226_004.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_005/20160226_005.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_006/20160226_006.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_007/20160226_007.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_008/20160226_008.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_009/20160226_009.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_010/20160226_010.avi'
#filename = '/home/cat/data/in_vivo/luis/20160226/M20160226_011/20160226_011.avi'

#M20160309
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_001/20160309_001.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_002/20160309_002.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_003/20160309_003.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_004/20160309_004.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_005/M20160309_005.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_006/20160309_006.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_007/20160309_007.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_008/20160309_008.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_009/20160309_009.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_010/20160309_010.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_011/20160309_011.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_012/20160309_012.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160309/M20160309_013/20160309_013.avi'

#M20160515
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_001/20160415_001.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_002/20160415_002.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_003/20160415_003.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_004/20160415_004.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_005/20160415_005.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_006/20160415_006.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_007/20160415_007.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_008/20160415_008.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160415/M20160415_009/20160415_009.avi'


#20160429
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_001/20160429_001.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_002/20160429_002.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_003/20160429_003.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_004/20160429_004.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_005/20160429_005.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_006/20160429_006.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_007/20160429_007.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_008/20160429_008.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_009/20160429_009.avi'
#filename = '/media/cat/250GB/in_vivo/luis/20160429/M20160429_010/20160429_010.avi'



#************************************************************************************************
#******************************** NEW LIST ******************************************************
#************************************************************************************************

#20160229
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_001/20160229_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_002/20160229_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_003/20160229_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_004/20160229_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_005/20160229_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_006/20160229_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_007/20160229_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_008/20160229_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_009/20160229_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_010/20160229_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_011/20160229_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_012/20160229_012.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_013/20160229_013.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_014/20160229_014.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_015/20160229_015.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_016/20160229_016.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_017/20160229_017.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_018/20160229_018.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_019/20160229_019.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160229/M20160229_020/20160229_020.avi'


#20160304
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_001/20160304_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_002/20160304_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_003/20160304_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_004/20160304_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_005/20160304_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_006/20160304_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_007/20160304_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_008/20160304_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_009/20160304_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_010/20160304_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_011/20160304_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_013/20160304_013.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_014/20160304_014.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_015/20160304_015.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_016/20160304_016.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_017/20160304_017.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_018/20160304_018.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160304/M20160304_019/20160304_019.avi'


#20160418
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_001/20160418_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_002/20160418_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_003/20160418_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_004/20160418_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_005/20160418_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_006/20160418_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_007/20160418_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_008/20160418_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_009/20160418_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160418/M20160418_010/20160418_010.avi'


##20160420
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_001/20160420_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_002/20160420_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_003/20160420_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_004/20160420_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_005/20160420_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_006/20160420_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_007/20160420_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_008/20160420_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_009/20160420_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_010/20160420_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_011/20160420_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_012/20160420_012.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_013/20160420_013.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_014/20160420_014.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_015/20160420_015.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_016/20160420_016.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_017/20160420_017.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160420/M20160420_018/20160420_018.avi'


##20160921 
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_001/20160921_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_002/20160921_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_003/20160921_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_004/20160921_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_005/20160921_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_006/20160921_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_007/20160921_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_008/20160921_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_009/20160921_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_010/20160921_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_011/20160921_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_012/20160921_012.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_013/20160921_013.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_014/20160921_014.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_015/20160921_015.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_016/20160921_016.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_017/20160921_017.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_018/20160921_018.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_019/20160921_019.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_020/20160921_020.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20160921/M20160921_021/20160921_021.avi'


##20161013
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161013/M20161013_001/20161013_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161013/M20161013_002/20161013_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161013/M20161013_003/20161013_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161013/M20161013_004/20161013_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161013/M20161013_005/20161013_005.avi'


##20161021
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_001/20161021_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_002/20161021_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_003/20161021_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_004/20161021_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_005/20161021_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_006/20161021_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_007/20161021_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_008/20161021_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_009/20161021_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_010/20161021_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_011/20161021_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_012/20161021_012.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_013/20161021_013.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_014/20161021_014.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_015/20161021_015.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_016/20161021_016.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161021/M20161021_017/20161021_017.avi'


##20161204
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_001/20161204_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_002/20161204_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_003/20161204_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_004/20161204_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_005/20161204_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_006/20161204_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_007/20161204_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_008/20161204_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_009/20161204_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_010/20161204_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_011/20161204_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_012/20161204_012.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_013/20161204_013.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_014/20161204_014.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161204/M20161204_015/20161204_015.avi'


##20161205
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_001/20161205_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_002/20161205_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_003/20161205_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_004/20161205_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_005/20161205_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_006/20161205_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_007/20161205_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_008/20161205_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_009/20161205_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_010/20161205_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_011/20161205_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161205/M20161205_012/20161205_012.avi'


##20161230
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_001/20161230_001.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_002/20161230_002.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_003/20161230_003.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_004/20161230_004.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_005/20161230_005.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_006/20161230_006.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_007/20161230_007.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_008/20161230_008.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_009/20161230_009.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_010/20161230_010.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_011/20161230_011.avi'
#filename = '/media/cat/2TB/in_vivo/luis/updated_video_list/20161230/M20161230_012/20161230_012.avi'


#****************load meta data ******************
root_dirs = [
'/media/cat/2TB/in_vivo/luis/updated_video_list/20160229',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20160304',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20160418',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20160420',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20160921',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20161013',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20161021',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20161024',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20161204',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20161205',
'/media/cat/2TB/in_vivo/luis/updated_video_list/20161230'
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

	#************************************************
	#******************* SET PARAMETERS *************
	#************************************************
	mot = MOTION(filename)		#Asigns mot.filename = filename
	mot.show_vid = False		#Flag to see videos during dim_reduction process
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
	    #mot.motion_correct_caiman()			#Caiman module for motion correction
	    #mot.binarize_frames()				#Reduce the video: none, contours, edges 
	    #mot.detect_movement()				#Detect movement by computing euclidean distance 
	    #mot.annotate_frames()				#Annotate movie frames; calls multiple functions	
	    #mot.recluster_id = 'rest'			#Re-annotate some of the frames clustered above
	    #mot.resplit_cluster(cluster=mot.recluster_id)			
	    #mot.save_metadata()
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
