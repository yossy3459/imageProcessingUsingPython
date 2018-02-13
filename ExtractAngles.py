# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import glob
import json

CV_WAITKEY_CURSORKEY_RIGHT  = 2555904; #右カーソルID
CV_WAITKEY_CURSORKEY_LEFT   = 2424832; #左カーソルID

# json parse
def jsonParse(fileName):

	# instantiate
	fileNum = 0
	jsonData = []

	file_list = sorted(glob.glob('./openpose/' + fileName + '/*.json'))

	for fileItor in file_list:
		with open(fileItor, 'r') as fileDesc:
			jsonData.append(json.load(fileDesc))
			# print('fileItor =' +  fileItor)
			print("frame{:03d}-RWrist: {}".format(fileNum, jsonData[fileNum]["people"][0]["pose_keypoints"][4]))
			print("frame{:03d}-RElbow: {}".format(fileNum, jsonData[fileNum]["people"][0]["pose_keypoints"][3]))

			fileNum += 1

	return jsonData

# json data
def ShowAngle(frame_W, frame_H, cap, jsonData):
	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)

	while (1):
		ret, frame = cap.read()
		if ret == False:
			break

		# カラー画像
		color_img = cv2.resize(frame, (frame_W // 2, frame_H // 2))

		# 画像表示
		cv2.imshow("ShowAngle", color_img)
		cv2.waitKey( 1 )

# save_video: ビデオの保存
def save_video( frame_W, frame_H, cap ) :

    cap.set( cv2.CAP_PROP_POS_FRAMES, 0 )

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    out    = cv2.VideoWriter( 'video_out.mp4' , fourcc, 20.0, (frame_W, frame_H) )
    #fourcc = -1
    #out = cv2.VideoWriter( 'video.avi' , fourcc, 20.0, (frame.shape[1], frame.shape[0]))

    count = 0
    while( True ) :
        ret, frame = cap.read()
        if ret == False :
            break

        count += 2
        cv2.circle(frame,(100+count,100), 50, (255,255,0),1)

        out.write( frame )

    out.release()





# main
if __name__ == '__main__':
	'''
	コマンドライン引数取得
	@param args[1] 取得する動画名
	'''
	args = sys.argv

	# read video
	videoName = args[1] + ".mp4"
	cap = cv2.VideoCapture(videoName)
	ret, frame = cap.read()

	frame_num  = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
	frame_H    = frame.shape[0]
	frame_W    = frame.shape[1]

	frame_half = cv2.resize(frame, (frame_W // 2, frame_H // 2) )
	cv2.imshow( "video vis", frame_half)
	frame_I    = 0

	# json parse
	jsonData = jsonParse(args[1])

	while (True) :
		key = cv2.waitKey( 0 )

		if(   key == ord('q') ) :
			exit()
		elif( key == CV_WAITKEY_CURSORKEY_RIGHT ) :
			frame_I = min(frame_I+1, frame_num-1)

		elif( key == CV_WAITKEY_CURSORKEY_LEFT  ) :
			frame_I = max(frame_I-1, 0)
		elif( key == ord('s') ) :
			save_video( frame_W, frame_H, cap )
		elif (key == ord('a')):
			ShowAngle(frame_H, frame_W, cap, jsonData)


		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)
		ret, frame = cap.read()
		frame_half = cv2.resize(frame, (frame_W // 2, frame_H // 2))
		cv2.imshow("video vis", frame_half)

		print("current frame i = ", frame_I)
