import numpy as np, cv2
from scipy import ndimage as ndi
import os
U=os.path.join(os.path.dirname(os.path.abspath(__file__)),'reference')+'/'
def fit_bg(img, mask_bg):
    h,w,_=img.shape; yy,xx=np.mgrid[0:h,0:w]; X=xx/w; Y=yy/h
    A=np.stack([np.ones_like(X),X,Y,X*X,Y*Y,X*Y,X**3,Y**3,X*X*Y,X*Y*Y],-1)
    M=A[mask_bg]; out=np.zeros_like(img,dtype=float)
    for c in range(3):
        coef,*_=np.linalg.lstsq(M,img[...,c][mask_bg],rcond=None); out[...,c]=A@coef
    return out
def segment(img, ymax=None, thr_l=13, thr_c=13, it=3):
    img=img.astype(float); h,w,_=img.shape
    bgm=np.zeros((h,w),bool); bgm[:, :12]=True; bgm[:, -12:]=True; bgm[:12,:]=True
    if ymax: bgm[ymax-30:ymax,:]=True
    for _ in range(it):
        bg=fit_bg(img,bgm)
        lab=cv2.cvtColor(np.clip(img,0,255).astype(np.uint8),cv2.COLOR_RGB2LAB).astype(float)
        labb=cv2.cvtColor(np.clip(bg,0,255).astype(np.uint8),cv2.COLOR_RGB2LAB).astype(float)
        dL=lab[...,0]-labb[...,0]; dC=np.hypot(lab[...,1]-labb[...,1],lab[...,2]-labb[...,2])
        fg=(dL>thr_l*2.55)|(dC>thr_c)|(dL<-60)
        if ymax: fg[ymax:]=False
        bgm=~ndi.binary_dilation(fg,iterations=6)
        if ymax: bgm[ymax:]=False
    fg=ndi.binary_opening(fg,iterations=1)
    lab_,n=ndi.label(fg); sizes=ndi.sum(fg,lab_,range(1,n+1))
    keep=np.zeros(n+1,bool); keep[1:]=sizes>max(60,sizes.max()*0.0015)
    fg=keep[lab_]
    fg=ndi.binary_closing(fg,iterations=2)
    fg=ndi.binary_fill_holes(fg)
    return fg,dL,dC

def refine(img, fg, ymax=None, iters=6):
    h,w=fg.shape
    m=np.full((h,w),cv2.GC_PR_BGD,np.uint8)
    near=ndi.binary_dilation(fg,iterations=25)
    m[~near]=cv2.GC_BGD
    m[fg]=cv2.GC_PR_FGD
    m[ndi.binary_erosion(fg,iterations=4)]=cv2.GC_FGD
    if ymax: m[ymax:]=cv2.GC_BGD
    bgd=np.zeros((1,65)); fgd=np.zeros((1,65))
    cv2.grabCut(np.ascontiguousarray(img[...,::-1]),m,None,bgd,fgd,iters,cv2.GC_INIT_WITH_MASK)
    out=(m==cv2.GC_FGD)|(m==cv2.GC_PR_FGD)
    lab_,n=ndi.label(out); sizes=ndi.sum(out,lab_,range(1,n+1))
    keep=np.zeros(n+1,bool); keep[1:]=sizes>max(60,sizes.max()*0.0015)
    out=ndi.binary_fill_holes(ndi.binary_opening(keep[lab_],iterations=1))
    return out
