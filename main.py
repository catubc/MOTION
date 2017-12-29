from MOTION import MOTION
filename ='/media/cat/4TB_ephys/in_vivo/james/ANM303491_Ph201/20151029-1425-A1/output.avi'

mot = MOTION.MOTION(filename)
mot.crop()
mot.dimreduce()
mot.detect_motion()
