import numpy as np

def get_zone15(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[ 0:60, 0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[0:60, 0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[0:60, 0:128])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[0:60, 0:160])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[ 0:60,160:256])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[ 0:60, 140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[ 0:60, 128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[ 0:60, 130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[ 0:60, 120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[ 0:60, 100:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[ 0:60, 75:256])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[ 0:60, 0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[ 0:60, 0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[ 0:60, 0:128])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[ 0:60, 0:128])
    return crops

def get_zone15_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[ 0:120, 0:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[0:120, 0:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[0:120, 0:256])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[0:120, 0:240])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[ 0:120,240:384])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[ 0:120, 280:512])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[ 0:120, 256:512])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[ 0:120, 260:512])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[ 0:120, 240:512])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[ 0:120, 200:512])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[ 0:120, 112:384])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[ 0:120, 0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[ 0:120, 0:180])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[ 0:120, 0:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[ 0:120, 0:256])
    return crops

def get_zone3(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 4:
            crops.append(np.array(img[slice_num])[200:315,55:205])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[200:315,0:200])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[200:315,0:155])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[200:315,0:130])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[200:315,0:110])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[200:315,0:110])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[200:315,0:110])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[200:315,150:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[200:315,150:256])
    return crops

def get_zone3_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[400:630,300:512])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[400:630,300:512])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[400:630,300:512])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[400:630,225:384])
        elif slice_num == 4:
            crops.append(np.array(img[slice_num])[400:630,55:205])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[400:630,0:300])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[400:630,0:310])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[400:630,0:260])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[400:630,0:220])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[400:630,0:220])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[400:630,0:220])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[400:630,225:384])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[400:630,300:512])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[400:630,300:512])
    return crops

def get_zone13(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[15:100,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[15:100,0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[15:100,0:148])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[15:100,0:190])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[15:100,160:256])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[15:100,140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[15:100,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[15:100,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[15:100,120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[15:100,80:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[15:100,35:256])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[15:100,0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[15:100,0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[15:100,0:128])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[15:100,0:128])
    return crops

#Incomplete
def get_zone13_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[30:200,0:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[30:200,0:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[30:200,0:296])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[30:200,0:285])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[30:200,240:384])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[30:200,280:512])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[30:200,256:512])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[30:200,260:512])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[30:200,240:512])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[30:200,160:512])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[30:200,52:384])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[30:200,0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[30:200,0:180])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[30:200,0:192])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[30:200,0:192])
    return crops

def get_zone11(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[40:150,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[40:150,0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[40:150,0:148])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[40:150,0:150])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[40:150,100:192])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[40:150,140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[40:150,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[40:150,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[40:150,120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[40:150,80:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[40:150,35:192])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[40:150,0:128])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[40:150,0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[40:150,0:138])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[40:150,0:138])
    return crops

def get_zone11_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[80:300,0:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[80:300,0:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[80:300,0:296])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[80:300,0:300])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[80:300,200:384])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[80:300,280:512])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[80:300,256:512])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[80:300,260:512])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[80:300,240:512])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[80:300,160:512])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[80:300,70:384])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[80:300,0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[80:300,0:240])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[80:300,0:276])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[80:300,0:276])
    return crops

def get_zone8(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[85:180,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[85:180,0:128])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[85:180,0:148])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[85:180,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[85:180,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[85:180,120:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[85:180,80:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[85:180,35:192])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[85:180,0:128])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[85:180,0:120])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[85:180,0:138])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[85:180,0:138])
    return crops

def get_zone8_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[170:360,0:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[170:360,0:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[170:360,0:296])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[170:360,256:512])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[170:360,260:512])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[170:360,240:512])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[170:360,160:512])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[170:360,70:384])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[170:360,0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[170:360,0:240])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[170:360,0:276])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[170:360,0:276])
    return crops

def get_zone6(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[130:230,0:128])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[130:230,0:128])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[130:230,100:192])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[130:230,140:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[130:230,128:256])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[130:230,130:256])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[130:230,105:256])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[130:230,65:256])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[130:230,0:192])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[130:230,0:128])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[130:230,0:150])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[130:230,0:150])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[130:230,0:138])
    return crops

def get_zone6_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[260:460,0:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[260:460,0:256])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[260:460,200:384])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[260:460,280:512])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[260:460,256:512])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[260:460,260:512])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[260:460,210:512])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[260:460,130:512])
        elif slice_num == 11:
            crops.append(np.array(img[slice_num])[260:460,0:384])
        elif slice_num == 12:
            crops.append(np.array(img[slice_num])[260:460,0:256])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[260:460,0:300])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[260:460,0:300])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[260:460,0:276])
    return crops

def get_zone4(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[245:330,150:256])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[245:330,150:256])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[245:330,135:256])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[245:330,20:192])
        elif slice_num == 4:
            crops.append(np.array(img[slice_num])[245:330,0:64])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[245:330,0:64])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[245:330,0:155])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[245:330,0:130])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[245:330,0:110])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[245:330,0:110])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[245:330,0:110])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[245:330,128:256])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[245:330,150:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[245:330,150:256])
    return crops

def get_zone4_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[490:660,300:512])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[490:660,300:512])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[490:660,270:512])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[490:660,40:384])
        elif slice_num == 4:
            crops.append(np.array(img[slice_num])[490:660,0:128])
        elif slice_num == 5:
            crops.append(np.array(img[slice_num])[490:660,0:128])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[490:660,0:310])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[490:660,0:260])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[490:660,0:220])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[490:660,0:220])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[490:660,0:220])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[490:660,256:512])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[490:660,300:512])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[490:660,300:512])
    return crops

# Note that this is not actually an accurate analog of the original get_zone9
# because when I wrote this function I didn't account for the x resize ratio on 
# some images
def get_zone9_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[160:380,160:352])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[160:380,140:352])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[160:380,90:352])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[160:380,30:352])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[160:380,160:512])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[160:380,160:372])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[160:380,160:352])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[160:380,80:352])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[160:380,0:312])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[160:380,160:432])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[160:380,160:372])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[160:380,160:352])
    return crops

def get_zone9(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[80:190,80:176])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[80:190,70:176])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[80:190,45:176])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[80:190,15:176])
        elif slice_num == 6:
            crops.append(np.array(img[slice_num])[80:190,80:256])
        elif slice_num == 7:
            crops.append(np.array(img[slice_num])[80:190,80:186])
        elif slice_num == 8:
            crops.append(np.array(img[slice_num])[80:190,80:176])
        elif slice_num == 9:
            crops.append(np.array(img[slice_num])[80:190,40:176])
        elif slice_num == 10:
            crops.append(np.array(img[slice_num])[80:190,0:156])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[80:190,80:216])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[80:190,80:186])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[80:190,80:176])
    return crops



def get_zone5(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[185:260,46:210])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[185:260,36:220])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[185:260,0:210])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[185:260,0:192])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[185:260,25:192])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[185:260,46:256])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[185:260,46:256])
    return crops

def get_zone5_fs(img):
    crops = []
    for slice_num in range(0, len(img)):
        if slice_num == 0:
            crops.append(np.array(img[slice_num])[370:520,92:420])
        elif slice_num == 1:
            crops.append(np.array(img[slice_num])[370:520,72:440])
        elif slice_num == 2:
            crops.append(np.array(img[slice_num])[370:520,0:420])
        elif slice_num == 3:
            crops.append(np.array(img[slice_num])[370:520,0:384])
        elif slice_num == 13:
            crops.append(np.array(img[slice_num])[370:520,50:384])
        elif slice_num == 14:
            crops.append(np.array(img[slice_num])[370:520,92:512])
        elif slice_num == 15:
            crops.append(np.array(img[slice_num])[370:520,92:512])
    return crops

def get_zone(tz, cropped_ims):
    if tz == 15:
        return get_zone15(cropped_ims)
    elif tz == 3:
        return get_zone3(cropped_ims)
    elif tz == 9:
        return get_zone9(cropped_ims)
    elif tz == 13:
        return get_zone13(cropped_ims)
    elif tz == 11:
        return get_zone11(cropped_ims)
    elif tz == 8:
        return get_zone8(cropped_ims)
    elif tz == 6:
        return get_zone6(cropped_ims)
    elif tz == 4:
        return get_zone4(cropped_ims)
    elif tz == 5:
        return get_zone5(cropped_ims)
    else:
        print ("Get zone FAILURE!")
        return None

def get_zone_fs(tz, cropped_ims):
    if tz == 9:
        return get_zone9_fs(cropped_ims)
    elif tz == 4:
        return get_zone4_fs(cropped_ims)
    elif tz == 3:
        return get_zone3_fs(cropped_ims)
    elif tz == 11:
        return get_zone11_fs(cropped_ims)
    elif tz == 5:
        return get_zone5_fs(cropped_ims)
    elif tz == 6:
        return get_zone6_fs(cropped_ims)
    elif tz == 15:
        return get_zone15_fs(cropped_ims)
    elif tz == 8:
        return get_zone8_fs(cropped_ims)
    elif tz == 13:
        return get_zone13_fs(cropped_ims)
    else:
        print ("Get zone FAILURE!")
        return None
