# -*- coding: utf-8 -*-
import sys
import cv2
import numpy as np
import math
import glob
import json
import os


# CV_WAITKEY_CURSORKEY_RIGHT = 2555904  # 右カーソルID
# CV_WAITKEY_CURSORKEY_LEFT = 2424832  # 左カーソルID
BRIGHTNESS_THRESHOLD = 200


'''
json_parse: jsonファイルの読み出しを行う
@param file_name 動画ファイル名(拡張子抜き) -> 対応するディレクトリからjsonを読み出す
@return json_data 取得したjson_dataの辞書
'''
def json_parse(file_name):

	# instantiate
	json_data = []

	file_list = sorted(glob.glob('./openpose/' + file_name + '/*.json'))

	for file_itor in file_list:
		with open(file_itor, 'r') as file_desc:
			json_data.append(json.load(file_desc))

	return json_data


'''
calculate_point_of_center: 線分abの中点を計算する
@param a 点(x, y)
@param b 点(x, y)
return 中点の座標
'''
def calculate_point_of_center(a, b):
	x = (b[0]+a[0]) / 2
	y = (b[1]+a[1]) / 2

	return (x, y)


'''
calculate_distance: 点と直線の距離を計算する
@param point 点(x, y)
@param line1_point 直線を与える点(x, y)
@param line2_point 直線を与える点(x, y)
return 距離
'''
def calculate_distance(point, line1_point, line2_point):
	u = np.array([line2_point[0]-line1_point[0], line2_point[1]-line1_point[1]])  # 直線
	v = np.array([line1_point[0]-point[0], line1_point[1]-point[1]])  # 点と直線のうちの一点

	# 外積を使った点と直線の距離
	distance = np.cross(u, v) / np.linalg.norm(u)

	return distance


'''
calculate_angle: 直線ab と 直線bc の角度を計算する
@param a 点(x, y)
@param b 点(x, y)、角度を図る角
@param c 点(x, y)
return 角度(degrees)
'''
def calculate_angle(a, b, c):
	# vector生成
	u = (a[0]-b[0], a[1]-b[1])  # ba
	v = (c[0]-b[0], c[1]-b[1])  # bc

	# 内積を用いたcosの計算
	cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

	# arccosを用いて角度(rad)の導出
	rad = math.acos(cos)

	# radをdegreesに変換
	return math.degrees(rad)


'''
labeling: ラベリング、人体の矩形領域と重心の検出、プレビュー表示
@param frame_H フレームの高さ
@param frame_W フレームの幅
@param cap 動画
@return rectangle ラベリング時に検出した人体の矩形領域
@return center_of_gravity ラベリング時に検出した人体の重心
'''
def labeling(frame_H, frame_W, cap):

	# initialize
	rectangle = []
	center_of_gravity = []
	frame_count = 0

	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)

	while (1):
		ret, frame = cap.read()
		if ret == False:
			break

		# カラー画像
		color_img = cv2.resize(frame, (frame_W // 2, frame_H // 2))

		# color threshold
		c_min = np.array([0, 0, 0], np.uint8)
		c_max = np.array([255, 255, BRIGHTNESS_THRESHOLD], np.uint8)

		frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
		binary_img = cv2.inRange(frame_hsv, c_min, c_max)
		binary_img = cv2.resize(binary_img, (frame_W // 2, frame_H // 2))

		#openningによるノイズ除去
		kernel = np.ones((3,3),np.uint8)
		binary_img = cv2.erode(binary_img, kernel,iterations = 1)
		binary_img = cv2.dilate(binary_img, kernel,iterations = 1)

		'''
		ラベリングの実行、その際に検出した矩形領域、中心座標を代入
		@param binary_img 2値画像
		@return labelnum ラベルのインデックス
		@return labelimg ラベリングした画像
		@return contours 矩形領域（x, y, width, height, size=面積)
		@return CoGs 重心座標(x,y)
		'''
		labelnum, labelimg, contours, CoGs = cv2.connectedComponentsWithStats(binary_img)

		# 各ラベルに対して
		for label in range(0, labelnum):

			# 指定した面積のラベルがあれば
			if (contours[:,4][label]>4096 and contours[:,4][label]<8192):

				# standard output
				print('frame: ' + '{:3}'.format(frame_count) + " -> rect: " + str(contours[label]) + ", CoGs: " + str(CoGs[label]))

				# 矩形領域と重心をカラー画像に書き込む
				x,y,w,h,size = contours[label]
				color_img = cv2.rectangle(color_img, (x,y), (x+w,y+h), (255,255,0), 1)
				x,y = CoGs[label]
				color_img = cv2.circle(color_img, (int(x),int(y)), 1, (255,255,0), 3)

				# 保存用
				rectangle.append(contours[label])
				center_of_gravity.append(CoGs[label])

		# 画像表示
		cv2.imshow("Extract Center of Gravity", color_img)
		cv2.waitKey(1)

		frame_count = frame_count + 1

	return rectangle, center_of_gravity


'''
extract: 各種解析用
@param frame_H フレームの高さ
@param frame_W フレームの幅
@param cap 動画
@param rectangle ラベリング時に検出した矩形領域
@param center_of_gravity ラベリング時に検出した重心
@param json_dot_data openposeで取得したjsonから取り出した点列
'''
def extract(frame_H, frame_W, cap, rectangle, center_of_gravity, json_dot_data, name):

	cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)

	# initialize
	# フレームのインデックスを管理する変数
	frame_count = 0

	# 保存先パス
	path_img1 = "./result/" + str(name) + "_1"
	path_img2 = "./result/" + str(name) + "_2"

	# ディレクトリ確認、なければmkdir
	if not os.path.exists(path_img1):
		os.makedirs(path_img1)
	if not os.path.exists(path_img2):
		os.makedirs(path_img2)

	while (1):
		ret, frame = cap.read()
		if ret == False:
			break

		try:
			'''
			json_dot_dataの取り出し
			'''
			neck = (json_dot_data[frame_count][1 * 3] / 2, json_dot_data[frame_count][1 * 3 + 1] / 2)
			sholder_right = (json_dot_data[frame_count][2 * 3] / 2, json_dot_data[frame_count][2 * 3 + 1] / 2)
			elbow_right = (json_dot_data[frame_count][3 * 3] / 2, json_dot_data[frame_count][3 * 3 + 1] / 2)
			wrist_right = (json_dot_data[frame_count][4 * 3] / 2, json_dot_data[frame_count][4 * 3 + 1] / 2)
			hip_right = (json_dot_data[frame_count][8 * 3] / 2, json_dot_data[frame_count][8 * 3 + 1] / 2)
			knee_right = (json_dot_data[frame_count][9 * 3] / 2, json_dot_data[frame_count][9 * 3 + 1] / 2)
			ankle_right = (json_dot_data[frame_count][10 * 3] / 2, json_dot_data[frame_count][10 * 3 + 1] / 2)
			hip_left = (json_dot_data[frame_count][11 * 3] / 2, json_dot_data[frame_count][11 * 3 + 1] / 2)
			knee_left = (json_dot_data[frame_count][12 * 3] / 2, json_dot_data[frame_count][12 * 3 + 1] / 2)
			ankle_left = (json_dot_data[frame_count][13 * 3] / 2, json_dot_data[frame_count][13 * 3 + 1] / 2)

			'''
			解析部
			'''
			# 腰の中心部の導出
			hip = calculate_point_of_center(hip_right, hip_left)

			# 重心と体の中心軸?(首と左右の腰の中点)の距離計算
			distance = calculate_distance(center_of_gravity[frame_count], neck, hip)
			print('frame:{0} -> distance:{1}'.format(frame_count, distance))

			# 角度計算
			# 右手首--右肩--右腰
			angle = calculate_angle(hip_right, sholder_right, wrist_right)
			print('frame:{0} -> angle:{1}'.format(frame_count, angle))

			# 右手首--右肘--右肩
			angle_1 = calculate_angle(wrist_right, elbow_right, sholder_right)
			print('frame:{0} -> angle_1:{1}'.format(frame_count, angle_1))
			# 右肘--右肩--右腰
			angle_2 = calculate_angle(elbow_right, sholder_right, hip_right)
			print('frame:{0} -> angle_2:{1}'.format(frame_count, angle_2))
			# 右腰--右膝--右足首
			angle_3 = calculate_angle(hip_right, knee_right, ankle_right)
			print('frame:{0} -> angle_3:{1}'.format(frame_count, angle_3))
			# 左腰--左膝--左足首
			angle_4 = calculate_angle(hip_left, knee_left, ankle_left)
			print('frame:{0} -> angle_4:{1}'.format(frame_count, angle_4))

			'''
			表示部
			'''
			color_img = cv2.resize(frame, (frame_W // 2, frame_H // 2))
			color_img2 = cv2.resize(frame, (frame_W // 2, frame_H // 2))

			# json_dataに従ってドットを打つ
			# img1, hip 以外
			for parts in (1, 2, 4, 8):
				x = json_dot_data[frame_count][parts * 3]//2
				y = json_dot_data[frame_count][parts * 3 + 1]//2
				color_img = cv2.circle(color_img, (int(x), int(y)), 1, (0, 0, 255), 3)

			# img2
			# angle_1: 右手首 -- 右肘 -- 右肩 (赤)
			for parts in (2, 3, 4):
				x = json_dot_data[frame_count][parts * 3]//2
				y = json_dot_data[frame_count][parts * 3 + 1]//2
				color_img2 = cv2.circle(color_img2, (int(x), int(y)), 2, (0, 0, 255), 3)
			# angle_2: 右肘 -- 右肩 -- 右腰 (緑)
			for parts in (2, 3, 8):
				x = json_dot_data[frame_count][parts * 3]//2
				y = json_dot_data[frame_count][parts * 3 + 1]//2
				color_img2 = cv2.circle(color_img2, (int(x), int(y)), 1, (0, 255, 0), 3)
			# angle_3: 右腰 - 右膝 - 右足首 (黄)
			for parts in (8, 9, 10):
				x = json_dot_data[frame_count][parts * 3]//2
				y = json_dot_data[frame_count][parts * 3 + 1]//2
				color_img2 = cv2.circle(color_img2, (int(x), int(y)), 1, (0, 255, 255), 3)
			# angle_4: 左腰 - 左膝 - 左足首 (青)
			for parts in (11, 12, 13):
				x = json_dot_data[frame_count][parts * 3]//2
				y = json_dot_data[frame_count][parts * 3 + 1]//2
				color_img2 = cv2.circle(color_img2, (int(x), int(y)), 1, (255, 0, 0), 3)

			# hip dot
			color_img = cv2.circle(color_img, (int(hip[0]), int(hip[1])), 1, (0, 0, 255), 3)

			# line
			# img1
			# 中心軸
			color_img = cv2.line(color_img, (int(neck[0]), int(neck[1])), (int(hip[0]), int(hip[1])), (0, 255, 0), 1);
			# angle
			color_img = cv2.line(color_img, (int(wrist_right[0]), int(wrist_right[1])), (int(sholder_right[0]), int(sholder_right[1])), (0, 255, 255), 1);
			color_img = cv2.line(color_img, (int(sholder_right[0]), int(sholder_right[1])), (int(hip_right[0]), int(hip_right[1])), (0, 255, 255), 1);

			# img2
			# angle_1
			color_img2 = cv2.line(color_img2, (int(wrist_right[0]), int(wrist_right[1])), (int(elbow_right[0]), int(elbow_right[1])), (0, 0, 255), 3);
			color_img2 = cv2.line(color_img2, (int(sholder_right[0]), int(sholder_right[1])), (int(elbow_right[0]), int(elbow_right[1])), (0, 0, 255), 3);
			# angle_2
			color_img2 = cv2.line(color_img2, (int(sholder_right[0]), int(sholder_right[1])), (int(elbow_right[0]), int(elbow_right[1])), (0, 255, 0), 1);
			color_img2 = cv2.line(color_img2, (int(sholder_right[0]), int(sholder_right[1])), (int(hip_right[0]), int(hip_right[1])), (0, 255, 0), 1);
			# angle_3
			color_img2 = cv2.line(color_img2, (int(hip_right[0]), int(hip_right[1])), (int(knee_right[0]), int(knee_right[1])), (0, 255, 255), 1);
			color_img2 = cv2.line(color_img2, (int(ankle_right[0]), int(ankle_right[1])), (int(knee_right[0]), int(knee_right[1])), (0, 255, 255), 1);
			# angle_4
			color_img2 = cv2.line(color_img2, (int(hip_left[0]), int(hip_left[1])), (int(knee_left[0]), int(knee_left[1])), (255, 0, 0), 1);
			color_img2 = cv2.line(color_img2, (int(ankle_left[0]), int(ankle_left[1])), (int(knee_left[0]), int(knee_left[1])), (255, 0, 0), 1);

			# ラベリングで検出した矩形、重心を表示
			x,y,w,h,size = rectangle[frame_count]
			color_img = cv2.rectangle(color_img, (x,y), (x+w,y+h), (255,255,0), 1)
			x,y = center_of_gravity[frame_count]
			color_img = cv2.circle(color_img, (int(x),int(y)), 2, (255,255,0), 3)

			# 文字表示
			# フォントの指定
			font = cv2.FONT_HERSHEY_DUPLEX
			# 文字の書き込み
			# img1
			text = 'frame: {:3}'.format(frame_count)
			color_img = cv2.putText(color_img, text, (5, 20), font, 0.6 ,(255, 255, 255))
			text = 'distance: {}'.format(distance)
			color_img = cv2.putText(color_img, text, (int(frame_W / 2)-320, int(frame_H / 2)-10), font, 0.6 ,(255, 255, 0))
			text = 'angle: {:14}'.format(angle)
			color_img = cv2.putText(color_img, text, (int(frame_W / 2)-320, int(frame_H / 2)-35), font, 0.6 ,(0, 255, 255))
			# img2
			text = 'frame: {:3}'.format(frame_count)
			color_img2 = cv2.putText(color_img2, text, (5, 20), font, 0.6 ,(255, 255, 255))
			text = 'angle1: {:14}'.format(angle_1)
			color_img2 = cv2.putText(color_img2, text, (int(frame_W / 2)-320, int(frame_H / 2)-85), font, 0.6 ,(0, 0, 255))
			text = 'angle2: {:14}'.format(angle_2)
			color_img2 = cv2.putText(color_img2, text, (int(frame_W / 2)-320, int(frame_H / 2)-60), font, 0.6 ,(0, 255, 0))
			text = 'angle3: {:14}'.format(angle_3)
			color_img2 = cv2.putText(color_img2, text, (int(frame_W / 2)-320, int(frame_H / 2)-35), font, 0.6 ,(0, 255, 255))
			text = 'angle4: {:14}'.format(angle_4)
			color_img2 = cv2.putText(color_img2, text, (int(frame_W / 2)-320, int(frame_H / 2)-10), font, 0.6 ,(255, 0, 0))

			# 画像表示
			cv2.imshow("Result", color_img)
			cv2.imshow("Result 2", color_img2)

			# ファイル名
			file_name_img1 = str(name) + "_1_" + "{:03}".format(frame_count) + ".png"
			file_name_img2 = str(name) + "_2_" + "{:03}".format(frame_count) + ".png"

			# イメージ書き込み
			cv2.imwrite(os.path.join(path_img1, file_name_img1), color_img)
			cv2.imwrite(os.path.join(path_img2, file_name_img2), color_img2)

			cv2.waitKey(1)

		except:
			print("exception occer")
			import traceback
			traceback.print_exc()
			break

		# フレームカウント インクリメント
		frame_count = frame_count + 1


'''
main
'''
if __name__ == '__main__':

	# initialize
	rectangle = []
	center_of_gravity = []
	json_dot_data = []
	out1 = []
	out2 = []

	'''
	コマンドライン引数取得
	@param args[1] 取得する動画名
	'''
	args = sys.argv

	# read video
	videoName = args[1] + ".mp4"

	cap = cv2.VideoCapture(videoName)
	ret, frame = cap.read()

	frame_num  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	frame_H    = frame.shape[0]
	frame_W    = frame.shape[1]

	frame_half = cv2.resize(frame, (frame_W // 2, frame_H // 2))
	cv2.imshow("video", frame_half)
	frame_I    = 0

	# json parse
	json_data = json_parse(args[1])
	print("Importing json_data: done!")

	for index in range(0, len(json_data)):
		json_dot_data.append(json_data[index]["people"][0]["pose_keypoints"])

	while (True) :
		print("Please input key...(q,x,z,l,e)")
		key = cv2.waitKey(0)

		if(key == ord('q')) :
			exit()
		elif(key == ord('x')):
			frame_I = min(frame_I+1, frame_num-1)
		elif(key == ord('z')):
			frame_I = max(frame_I-1, 0)

		# labeling
		elif(key == ord('l')):
			rectangle, center_of_gravity = labeling(frame_H, frame_W, cap)

		# extract_angles
		elif(key == ord('e')):
			extract(frame_H, frame_W, cap, rectangle, center_of_gravity, json_dot_data, args[1])


		cap.set(cv2.CAP_PROP_POS_FRAMES, frame_I)
		ret, frame = cap.read()
		frame_half = cv2.resize(frame, (frame_W // 2, frame_H // 2))
		cv2.imshow("video", frame_half)

		print("current frame i = ", frame_I)
