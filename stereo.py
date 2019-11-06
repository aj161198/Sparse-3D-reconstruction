import numpy as np
import cv2
from matplotlib import pyplot as plt


def drawlines(imgL,imgR,lines,cornersL,cornersR):
	r,c = imgL.shape
	imgL = cv2.cvtColor(imgL,cv2.COLOR_GRAY2BGR)
	imgR = cv2.cvtColor(imgR,cv2.COLOR_GRAY2BGR)
	for r,pt1,pt2 in zip(lines,cornersL,cornersR):
		color = tuple(np.random.randint(0,255,3).tolist())
		
		x0,y0 = map(int, [0, -r[2]/r[1] ])
		x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
		cv2.line(imgL, (x0,y0), (x1,y1), color,1)
		cv2.circle(imgL, (pt1[0][0], pt1[0][1]), 5, color, -1)
		cv2.circle(imgR, (pt2[0][0], pt2[0][1]), 5, color, -1)
	return imgL,imgR

def create_output(vertices, colors, filename):
	colors = colors.reshape(-1,3)
	vertices = np.hstack([vertices.reshape(-1,3),colors])

	ply_header = '''ply
		format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar red
		property uchar green
		property uchar blue
		end_header
		'''
	with open(filename, 'w') as f:
		f.write(ply_header %dict(vert_num=len(vertices)))
		np.savetxt(f,vertices,'%f %f %f %d %d %d')


K = np.array( [[633.83454283,   0.,         291.92093036],
 [  0.,         628.63790165, 235.70267598],
 [  0.,           0.,           1.        ]])

D = np.array( [[ 0.,  0., 0, 0,  0.]])

img1 = cv2.imread('/home/aman/Documents/0.jpg',0) 
img2 = cv2.imread('/home/aman/Documents/1.jpg',0) 

# Initiate SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test
good = []
for m,n in matches:
	if m.distance < 0.70*n.distance:
		good.append(m)

goods = []
for m,n in matches:
	if m.distance < 0.70*n.distance:
		goods.append([m])

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None)

plt.imshow(img3, 'gray'),plt.show()

flag = -1
if len(good)>10:
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	E, mask = cv2.findEssentialMat(src_pts, dst_pts, K)
	src_pts = src_pts[mask==1].T
	dst_pts = dst_pts[mask==1].T
	R1,R2,t = cv2.decomposeEssentialMat(E)
	R = np.array([R1,R1,R2,R2])
	T = np.array([t,-t,t,-t])
	
	P1 = np.zeros((3,4))
	P1[:,:3] = np.eye(3)
	P1 = K @ P1
	for i in range(4):
		P2 = np.zeros((3,4))
		P2[:,:3] = R[i]
		P2[:, 3] = T[i].reshape(3,)
		P2 = K @ P2
		points_4D = cv2.triangulatePoints(P1, P2, src_pts, dst_pts)
		X, Y, Z = points_4D[:3, :] / points_4D[3, :]
		Z_ = R[i][2,0] * X + R[i][2,1] * Y + R[i][2,2] * Z + T[i][2] 
		if len(np.where(Z<0)[0]) == 0 and len(np.where(Z_<0)[0]) == 0:
			flag = i
			break
	if flag >= 0:
		R,T = R[flag], T[flag]
		R1, R2, P1, P2, Q, a, b = cv2.stereoRectify(K, D, K, D, (640,480), R, T, flags = 0)
		map1, map2 = cv2.initUndistortRectifyMap(K, D, R1, P1, (640,480), cv2.CV_16SC2)
		imgL = cv2.remap(img1, map1, map2,  cv2.INTER_CUBIC)

		map3, map4 = cv2.initUndistortRectifyMap(K, D, R2, P2, (640,480), cv2.CV_16SC2)
		imgR = cv2.remap(img2, map3, map4, cv2.INTER_CUBIC)

		cv2.imshow("1", imgL)
		cv2.imshow("2", imgR)
		cv2.waitKey(0)
		a1,a2,a3 = P2[:,3]
		A = np.array([[0,-a3,a2],[a3,0,-a1],[-a2,a1,0]], np.float64)

		PT = np.zeros((4,3))
		PT[:3,:] = np.linalg.inv(K)

		F = A @ P2 @ PT
		F = F / F[2][2]

		"""criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
		ret, cornersL = cv2.findChessboardCorners(imgL, (9,7),None)
		cornersL = cv2.cornerSubPix(imgL, cornersL, (11, 11), (-1, -1), criteria)
		if cornersL[0][0][0] > cornersL[8][0][0]:
			cornersL = cornersL[::-1]

		ret, cornersR = cv2.findChessboardCorners(imgR, (9,7),None)
		cornersR = cv2.cornerSubPix(imgR, cornersR, (11, 11), (-1, -1), criteria)
		if cornersR[0][0][0] > cornersR[8][0][0]:
			cornersR = cornersR[::-1]


		lines1 = cv2.computeCorrespondEpilines(cornersR.reshape(-1,1,2), 2,F)
		lines1 = lines1.reshape(-1,3)
		img5,img6 = drawlines(imgL,imgR,lines1,cornersL,cornersR)

		lines2 = cv2.computeCorrespondEpilines(cornersL.reshape(-1,1,2), 1,F)
		lines2 = lines2.reshape(-1,3)
		img3,img4 = drawlines(imgR,imgL,lines2,cornersR,cornersL)

		cv2.imshow("1", img3)
		cv2.imshow("2", img4)
		cv2.waitKey(0)"""

		
		window_size = 3                     
		left_matcher = cv2.StereoSGBM_create(
		    minDisparity=0,
		    numDisparities=32,             
		    blockSize=5,
		    P1=8 * 3 * window_size ** 2,   
		    P2=32 * 3 * window_size ** 2,
		    disp12MaxDiff=1,
		    uniquenessRatio=15,
		    speckleWindowSize=0,
		    speckleRange=2,
		    preFilterCap=63,
		    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
		)

		right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
		lmbda = 80000
		sigma = 1.2
		visual_multiplier = 1.0
		 
		wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
		wls_filter.setLambda(lmbda)
		wls_filter.setSigmaColor(sigma)

		print('computing disparity...')
		displ = left_matcher.compute(imgL, imgR)  # .astype(np.float32)/16
		dispr = right_matcher.compute(imgR, imgL)  # .astype(np.float32)/16
		displ = np.int16(displ)
		dispr = np.int16(dispr)
		filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!

		plt.imshow(filteredImg, 'gray')
		plt.show()

		img = cv2.imread('/home/aman/Documents/0.jpg')		
		img = cv2.remap(img, map1, map2,  cv2.INTER_CUBIC)

		mask = filteredImg > 0
		_img3D = cv2.reprojectImageTo3D(filteredImg, Q)
		colors = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		points = _img3D[mask == 1]
		colors = colors[mask == 1]
		create_output(points, colors, "stereo.ply")


