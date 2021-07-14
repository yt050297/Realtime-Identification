from module.camera_manager import CameraManager
from module.camera_manager import TriggerType
from module.camera_manager import AcquisitionMode
from module.camera_manager import AutoExposureMode
from module.camera_manager import AutoGainMode
from module.imread_imwrite_japanese import ImreadImwriteJapanese
from module.create_reference import CreateReference
from module.fwhm import FWHM
import cv2
import time
import numpy as np
import os
from datetime import datetime as dt
import glob
from statistics import mode
from scipy import stats


class ShowInfraredCamera():
    def __init__(self):
        self.cam_manager = CameraManager()
        self.initsavecount = 0
        self.savecount = 0
        self.existnumber = 0
        self.colormap_table_count = 0
        self.colormap_table = [
            ['COLORMAP_JET', cv2.COLORMAP_JET],
            ['COLORMAP_AUTUMN', cv2.COLORMAP_AUTUMN],
            ['COLORMAP_BONE', cv2.COLORMAP_BONE],
            ['COLORMAP_COOL', cv2.COLORMAP_COOL],
            ['COLORMAP_HOT', cv2.COLORMAP_HOT],
            ['COLORMAP_HSV', cv2.COLORMAP_HSV],
            ['COLORMAP_OCEAN', cv2.COLORMAP_OCEAN],
            ['COLORMAP_PINK', cv2.COLORMAP_PINK],
            ['COLORMAP_RAINBOW', cv2.COLORMAP_RAINBOW],
            ['COLORMAP_SPRING', cv2.COLORMAP_SPRING],
            ['COLORMAP_SUMMER', cv2.COLORMAP_SUMMER],
            ['COLORMAP_WINTER', cv2.COLORMAP_WINTER],
        ]
        self.norm = False
        self.detectflag = 0
        self.video_saveflag = False
        self.im_jp = ImreadImwriteJapanese
        #変更点
        self.list_1 = []
        self.list_2 = []
        self.list_3 = []
        self.list_4 = []
        self.list_5 = []
        self.list_6 = []
        self.list_7 = []
        self.flag = 0
        self.flag2 = 0
        self.tag = 0
        self.background_width = 1440
        self.background_height = 540


    def beam_profiler(self, trigger, gain, exp, flip):

        if trigger == "software":
            self.cam_manager.choose_trigger_type(TriggerType.SOFTWARE)
        elif trigger == "hardware":
            self.cam_manager.choose_trigger_type(TriggerType.HARDWARE)

        self.cam_manager.turn_on_trigger_mode()

        self.cam_manager.choose_acquisition_mode(AcquisitionMode.CONTINUOUS)

        self.cam_manager.choose_auto_exposure_mode(AutoExposureMode.OFF)
        self.cam_manager.set_exposure_time(exp)

        self.cam_manager.choose_auto_gain_mode(AutoGainMode.OFF)
        self.cam_manager.set_gain(gain)

        self.cam_manager.start_acquisition()

        self.create_reference = CreateReference()
        self.fwhm = FWHM()

        while True:
            # 処理前の時刻
            t1 = time.time()
            if trigger == "software":
                self.cam_manager.execute_software_trigger()

            frame = self.cam_manager.get_next_image()
            if frame is None:
                continue

            if self.norm == True:
                frame = self.min_max_normalization(frame)

            if flip == 'normal':
                pass
            elif flip == 'flip':
                frame = cv2.flip(frame, 1)  # 画像を左右反転

            if self.detectflag == 1:
                ellipses = self.create_reference.realtime_create_reference(frame, self.numbeams, self.minsize,
                                                                           self.maxsize, self.binthresh)
                if len(ellipses) == self.numbeams:
                    frame = self.fwhm.realtime_fwhm(frame, ellipses)
                    self.detectflag = 2
                else:
                    print('ビームがうまく検出されませんでした。設定を見返して下さい。')
                    self.detectflag = 0
            elif self.detectflag == 2:
                frame = self.fwhm.realtime_fwhm(frame, ellipses)

            cv2.imshow("Please push Q button when you want to close the window.", cv2.resize(frame, (800, 800)))

            if self.initsavecount == 0 and self.savecount == 0:
                pass
            elif self.initsavecount != self.savecount:
                if os.path.exists(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber)):
                    for existnumber in range(len(glob.glob(self.savepath + '/*.png'))):
                        self.existnumber = existnumber + 1
                    print('同じファイルが存在しているので、ファイルを新規作成します')
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       frame)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
                else:
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       frame)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
            elif self.initsavecount == self.savecount:
                self.initsavecount = self.savecount = 0
                self.existnumber = 0
                print('Initialize savecount')

            if self.video_saveflag == True:
                self.out.write(cv2.resize(frame, (self.width, self.height)))

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                if self.video_saveflag == True:
                    self.out.release()
                    self.video_saveflag = False
                    print('録画終了')
                cv2.destroyAllWindows()
                print('Complete Cancel')
                break

            # 処理後の時刻
            t2 = time.time()

            try:
                freq = 1 / (t2 - t1)
                print(f"フレームレート：{freq}fps")
            except ZeroDivisionError:
                pass

        self.cam_manager.stop_acquisition()

    def beam_profiler_color(self, trigger, gain, exp, flip):

        if trigger == "software":
            self.cam_manager.choose_trigger_type(TriggerType.SOFTWARE)
        elif trigger == "hardware":
            self.cam_manager.choose_trigger_type(TriggerType.HARDWARE)

        self.cam_manager.turn_on_trigger_mode()

        self.cam_manager.choose_acquisition_mode(AcquisitionMode.CONTINUOUS)

        self.cam_manager.choose_auto_exposure_mode(AutoExposureMode.OFF)
        self.cam_manager.set_exposure_time(exp)

        self.cam_manager.choose_auto_gain_mode(AutoGainMode.OFF)
        self.cam_manager.set_gain(gain)

        self.cam_manager.start_acquisition()

        self.create_reference = CreateReference()
        self.fwhm = FWHM()

        while True:
            # 処理前の時刻
            t1 = time.time()
            if trigger == "software":
                self.cam_manager.execute_software_trigger()

            frame = self.cam_manager.get_next_image()
            if frame is None:
                continue

            if self.norm == True:
                frame = self.min_max_normalization(frame)

            if flip == 'normal':
                pass
            elif flip == 'flip':
                frame = cv2.flip(frame, 1)  # 画像を左右反転

            # 疑似カラーを付与
            apply_color_map_image = cv2.applyColorMap(frame, self.colormap_table[
                self.colormap_table_count % len(self.colormap_table)][1])

            if self.detectflag == 1:
                ellipses = self.create_reference.realtime_create_reference(frame, self.numbeams, self.minsize,
                                                                           self.maxsize, self.binthresh)
                if len(ellipses) == self.numbeams:
                    apply_color_map_image = self.fwhm.realtime_fwhm(apply_color_map_image, ellipses)
                    self.detectflag = 2
                else:
                    print('ビームがうまく検出されませんでした。設定を見返して下さい。')
                    self.detectflag = 0
            elif self.detectflag == 2:
                apply_color_map_image = self.fwhm.realtime_fwhm(apply_color_map_image, ellipses)

            cv2.putText(apply_color_map_image,
                        self.colormap_table[self.colormap_table_count % len(self.colormap_table)][0],
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

            cv2.imshow("Please push Q button when you want to close the window.",
                       cv2.resize(apply_color_map_image, (800, 800)))

            if self.initsavecount == 0 and self.savecount == 0:
                pass
            elif self.initsavecount != self.savecount:
                if os.path.exists(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber)):
                    for existnumber in range(len(glob.glob(self.savepath + '/*.png'))):
                        self.existnumber = existnumber + 1
                    print('同じファイルが存在しているので、ファイルを新規作成します')
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       apply_color_map_image)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
                else:
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       apply_color_map_image)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
            elif self.initsavecount == self.savecount:
                self.initsavecount = self.savecount = 0
                self.existnumber = 0
                print('Initialize savecount')

            if self.video_saveflag == True:
                self.out.write(cv2.resize(apply_color_map_image, (self.width, self.height)))

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                if self.video_saveflag == True:
                    self.out.release()
                    self.video_saveflag = False
                    print('録画終了')
                cv2.destroyAllWindows()
                print('Complete Cancel')
                break

            elif k == ord('n'):  # N
                self.colormap_table_count = self.colormap_table_count + 1
            # 処理後の時刻
            t2 = time.time()

            # 経過時間を表示
            freq = 1 / (t2 - t1)
            print(f"フレームレート：{freq}fps")

        self.cam_manager.stop_acquisition()

    def realtime_identification(self, classnamelist, model, trigger, gain, exp, im_size_width, im_size_height, flip):

        if trigger == "software":
            self.cam_manager.choose_trigger_type(TriggerType.SOFTWARE)
        elif trigger == "hardware":
            self.cam_manager.choose_trigger_type(TriggerType.HARDWARE)

        self.cam_manager.turn_on_trigger_mode()

        self.cam_manager.choose_acquisition_mode(AcquisitionMode.CONTINUOUS)

        self.cam_manager.choose_auto_exposure_mode(AutoExposureMode.OFF)
        self.cam_manager.set_exposure_time(exp)

        self.cam_manager.choose_auto_gain_mode(AutoGainMode.OFF)
        self.cam_manager.set_gain(gain)

        self.cam_manager.start_acquisition()

        font = cv2.FONT_HERSHEY_PLAIN
        fontsize = 6
        samplename_position_x = probability_position_x = 90
        samplename_position_y = 120
        probability_position_y = 220
        x_move = 800
        font_scale = 5
        while True:
            # 処理前の時刻
            t1 = time.time()
            if trigger == "software":
                self.cam_manager.execute_software_trigger()

            frame = self.cam_manager.get_next_image()
            if frame is None:
                continue
            # 読み込んだフレームを書き込み
            if self.norm == True:
                frame = self.min_max_normalization(frame)

            if flip == 'normal':
                pass
            elif flip == 'flip':
                frame = cv2.flip(frame, 1)  # 画像を左右反転

            resize_image = cv2.resize(frame, (im_size_width, im_size_height))
            # print(resize_image)
            # print('writing')
            X = []
            X.append(resize_image)
            X = np.array(X)
            X = X.astype("float") / 256

            X.resize(X.shape[0], X.shape[1], X.shape[2], 1)

            predict = model.predict(X)

            for (i, pre) in enumerate(predict):
                y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル

                cv2.putText(frame, 'Predict sample', (samplename_position_x, samplename_position_y), font, fontsize,
                            (255, 255, 255), font_scale, cv2.LINE_AA)
                cv2.putText(frame, 'Probability', (probability_position_x, probability_position_y), font, fontsize,
                            (255, 255, 255), font_scale, cv2.LINE_AA)
                pretext = classnamelist[y]
                cv2.putText(frame, pretext, (samplename_position_x + x_move, samplename_position_y), font, fontsize,
                            (255, 255, 255), font_scale, cv2.LINE_AA)

                cv2.putText(frame, '{}%'.format(round(pre[y] * 100)),
                            (probability_position_x + x_move, probability_position_y), font, fontsize,
                            (255, 255, 255), font_scale, cv2.LINE_AA)

            cv2.imshow("Please push Q button when you want to close the window.",
                       cv2.resize(frame, (800, 800)))

            if self.initsavecount == 0 and self.savecount == 0:
                pass
            elif self.initsavecount != self.savecount:
                if os.path.exists(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber)):
                    for existnumber in range(len(glob.glob(self.savepath + '/*.png'))):
                        self.existnumber = existnumber + 1
                    print('同じファイルが存在しているので、ファイルを新規作成します')
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       frame)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
                else:
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       frame)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
            elif self.initsavecount == self.savecount:
                self.initsavecount = self.savecount = 0
                self.existnumber = 0
                print('Initialize savecount')

            if self.video_saveflag == True:
                self.out.write(cv2.resize(frame, (self.width, self.height)))

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                if self.video_saveflag == True:
                    self.out.release()
                    self.video_saveflag = False
                    print('録画終了')
                cv2.destroyAllWindows()
                print('Complete Cancel')
                break

            # 処理後の時刻
            t2 = time.time()

            # 経過時間を表示
            try:
                freq = 1 / (t2 - t1)
                print(f"フレームレート：{freq}fps")
            except ZeroDivisionError:
                pass

        self.cam_manager.stop_acquisition()
        print('Stopped Camera')



    def realtime_identification_color(self, classnamelist, model, trigger, gain, exp, im_size_width, im_size_height,
                                      flip):
        def write_backgrand_imaging(height, width):
            img = np.zeros((height, width, 3), np.uint8)
            return img

        def write_onepixel_imaging(img, fromleft, fromupper, pixel_size, color, width):
            img = cv2.rectangle(img, (fromleft, fromupper), (fromleft + pixel_size, fromupper + width), color, -1)
            return img

        def color_append(pre_num, imaging, fromleft, fromupper, pixel_size, vertical):
            if pre_num == 4:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper - 20, 10, (255, 255, 255), vertical + 40)
            elif pre_num == 0:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size, (0, 0, 255), vertical)
            elif pre_num == 1:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size, (0, 255, 0), vertical)
            elif pre_num == 2:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size, (255, 0, 0), vertical)
            elif pre_num == 3:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, pixel_size, (0, 255, 255), vertical)
            return imaging

        def count_accuracy(num, list):
            pre, count = stats.mode(list)
            all_frame_num = len(list)
            acc_pre_frame_num = list.count(pre[0])
            ave_acc = (acc_pre_frame_num / all_frame_num) * 100
            print('tag_num : {}'.format(num))
            print('list : {}\n'.format(list))
            print('average_predict : {}\n'.format(pre[0]))
            print('all_frame_number : {}\n'.format(all_frame_num))
            print('accurate_predict_frame_number : {}\n'.format(acc_pre_frame_num))
            print('average_accuracy:{}%\n'.format(ave_acc))
            return pre[0]


        if trigger == "software":
            self.cam_manager.choose_trigger_type(TriggerType.SOFTWARE)
        elif trigger == "hardware":
            self.cam_manager.choose_trigger_type(TriggerType.HARDWARE)

        self.cam_manager.turn_on_trigger_mode()

        self.cam_manager.choose_acquisition_mode(AcquisitionMode.CONTINUOUS)

        self.cam_manager.choose_auto_exposure_mode(AutoExposureMode.OFF)
        self.cam_manager.set_exposure_time(exp)

        self.cam_manager.choose_auto_gain_mode(AutoGainMode.OFF)
        self.cam_manager.set_gain(gain)

        self.cam_manager.start_acquisition()

        font = cv2.FONT_HERSHEY_PLAIN
        fontsize = 6
        samplename_position_x = probability_position_x = 90
        samplename_position_y = 120
        probability_position_y = 220
        x_move = 800
        #変更点
        y_move = 100
        font_scale = 5

        #####ここからは、タグイメージング用のパラメーター
        fontsize_tag = 1.3
        fromleft = 10  # 最初のピクセルの左端からの位置
        fromupper = 270  # 最初のピクセルの上端からの位置
        vertical = 150  # imagingの際の縦の長さ
        # background_width = 1440
        # background_height = 540
        pre_name = 'Predict Tag : '
        #####ここから変える必要あり
        #fps = 10
        pixel_size = 50  # 移動距離＆1pixelのサイズ
        # result_imaging = 'C:/Users/yt050/Desktop/saveimaging/tag_imaging.mp4'

        imaging = write_backgrand_imaging(self.background_height, self.background_width)  # タグイメージングのための背景を記入
        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2, cv2.LINE_AA)  # 文字記入
        #cv2.imshow('predict tag imaging', imaging)

        while True:
            # 処理前の時刻
            t1 = time.time()
            if trigger == "software":
                self.cam_manager.execute_software_trigger()

            frame = self.cam_manager.get_next_image()
            if frame is None:
                continue

            if self.norm == True:
                frame = self.min_max_normalization(frame)

            # 読み込んだフレームを書き込み
            if flip == 'normal':
                pass
            elif flip == 'flip':
                frame = cv2.flip(frame, 1)  # 画像を左右反転

            resize_image = cv2.resize(frame, (im_size_width, im_size_height))

            X = []
            X.append(resize_image)
            X = np.array(X)
            X = X.astype("float") / 256

            X.resize(X.shape[0], X.shape[1], X.shape[2], 1)

            predict = model.predict(X)

            # 疑似カラーを付与
            apply_color_map_image = cv2.applyColorMap(frame, self.colormap_table[
                self.colormap_table_count % len(self.colormap_table)][1])

            for (i, pre) in enumerate(predict):

                z = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル



                ########################################## 変更点_カードリーダー方式用  #####################################

                ##ラベル合わせ
                if z == 0 or z == 1:
                    y = 0
                elif z == 2 or z == 3:
                    y = 1
                elif z == 4 or z == 5:
                    y = 2
                elif z == 6 or z == 7:
                    y = 3
                elif z == 8:
                    y = 4
                else:
                    pass

                # １つ目のタグ
                if self.flag == 0:
                    if y == 4:
                        self.flag = 1
                    if y != 4:
                        pass

                if self.flag == 1:
                    if y != 4:
                        self.list_1.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        self.flag2 = 1
                    if y == 4 and self.flag2 == 1:
                        self.flag = 2
                    else:
                        pass

                # ２回目のNoneのとき
                if self.flag == 2:
                    if y != 4:
                        pass
                    if y == 4:
                        #pre_1 = mode(self.list_1)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + 10
                        pre_1 = count_accuracy(1, self.list_1)
                        pre_name = pre_name + str(pre_1)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1)
                        #print('tag{}'.format(pre_1))
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 3

                # ２つ目のタグ
                if self.flag == 3:
                    if y != 4:
                        self.list_2.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag2 = 2
                    if y == 4 and self.flag2 == 2:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 4
                    else:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                # ３回目のNoneのとき
                if self.flag == 4:
                    if y != 4:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    if y == 4:
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + 10
                        #pre_2 = mode(self.list_2)
                        pre_2 = count_accuracy(2, self.list_2)
                        pre_name = pre_name + str(pre_2)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1) + str(pre_2)
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 5

                # ３つ目のタグ
                if self.flag == 5:
                    if y != 4:
                        self.list_3.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag2 = 3
                    if y == 4 and self.flag2 == 3:
                        self.flag = 6
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    else:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                # 4回目のNoneのとき
                if self.flag == 6:
                    if y != 4:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    if y == 4:
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + 10
                        # pre_2 = mode(self.list_2)
                        pre_3 = count_accuracy(3, self.list_3)
                        pre_name = pre_name + str(pre_3)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1) + str(pre_2)+str(pre_3)
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 7

                # 4つ目のタグ
                if self.flag == 7:
                    if y != 4:
                        self.list_4.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag2 = 4
                    if y == 4 and self.flag2 == 4:
                        self.flag = 8
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    else:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                # 5回目のNoneのとき
                if self.flag == 8:
                    if y != 4:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    if y == 4:
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + 10
                        # pre_2 = mode(self.list_2)
                        pre_4 = count_accuracy(4, self.list_4)
                        pre_name = pre_name + str(pre_4)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1) + str(pre_2) + str(pre_3) + str(pre_4)
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 9

                # 5つ目のタグ
                if self.flag == 9:
                    if y != 4:
                        self.list_5.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag2 = 5
                    if y == 4 and self.flag2 == 5:
                        self.flag = 10
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    else:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                 # 6回目のNoneのとき
                if self.flag == 10:
                    if y != 4:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    if y == 4:
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + 10
                        # pre_2 = mode(self.list_2)
                        pre_5 = count_accuracy(5, self.list_5)
                        pre_name = pre_name + str(pre_5)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1) + str(pre_2) +str(pre_3) + str(pre_4) + str(pre_5)
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 11

                # 6つ目のタグ
                if self.flag == 11:
                    if y != 4:
                        self.list_6.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag2 = 6
                    if y == 4 and self.flag2 == 6:
                        self.flag = 12
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    else:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                 # 7回目のNoneのとき
                if self.flag == 12:
                    if y != 4:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    if y == 4:
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + 10
                        # pre_2 = mode(self.list_2)
                        pre_6 = count_accuracy(6, self.list_6)
                        pre_name = pre_name + str(pre_6)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1) + str(pre_2) + str(pre_3) + str(pre_4) + str(pre_5) +str(pre_6)
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 13

                # 7つ目のタグ
                if self.flag == 13:
                    if y != 4:
                        self.list_7.append(y)
                        color_append(y, imaging, fromleft, fromupper, pixel_size, vertical)
                        fromleft = fromleft + pixel_size
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag2 = 7
                    if y == 4 and self.flag2 == 7:
                        self.flag = 14
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                    else:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                # 測定終了(8回目のNone)
                if self.flag == 14:
                    if y != 4:
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (0, 255, 255), font_scale, cv2.LINE_AA)
                    if y == 4:
                        #pre_3 = mode(self.list_3)
                        pre_7 = count_accuracy(7, self.list_7)
                        pre_name = pre_name + str(pre_7)
                        imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (255, 255, 255), 2,
                                              cv2.LINE_AA)  # 文字記入
                        self.tag = str(pre_1) + str(pre_2) + str(pre_3) + str(pre_4) + str(pre_5) + str(pre_6) + str(pre_7)
                        cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                    (probability_position_x, probability_position_y + y_move),
                                    font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                        self.flag = 15

                if self.flag == 15:
                    cv2.putText(apply_color_map_image, 'Predict tag: {}'.format(self.tag),
                                (probability_position_x, probability_position_y + y_move),
                                font, fontsize, (0, 165, 255), font_scale, cv2.LINE_AA)
                    imaging = cv2.putText(imaging, pre_name, (10, 200), 2, fontsize_tag, (0, 165, 255), 2,
                                          cv2.LINE_AA)  # 文字記入


                # # 初期に戻すのとなんのタグがを表示
                # if self.flag == 15:
                #     if self.tag == '302':
                #         tag_name = 'A'
                #         cv2.putText(apply_color_map_image, 'Predict tag: {0}/Identification: {1}'.format(self.tag,tag_name),
                #                     (probability_position_x, probability_position_y + y_move),
                #                     font, fontsize, (0, 255, 255), font_scale, cv2.LINE_AA)
                #     if self.tag == '103':
                #         tag_name = 'B'
                #         cv2.putText(apply_color_map_image, 'Predict tag: {0}/Identification: {1}'.format(self.tag,tag_name),
                #                     (probability_position_x, probability_position_y + y_move),
                #                     font, fontsize, (0, 255, 255), font_scale, cv2.LINE_AA)
                #     if self.tag == '021':
                #         tag_name = 'C'
                #         cv2.putText(apply_color_map_image,
                #                     'Predict tag: {0}/Identification: {1}'.format(self.tag, tag_name),
                #                     (probability_position_x, probability_position_y + y_move),
                #                     font, fontsize, (0, 255, 255), font_scale, cv2.LINE_AA)


                cv2.putText(apply_color_map_image, 'Predict sample', (samplename_position_x, samplename_position_y),
                            font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                cv2.putText(apply_color_map_image, 'Probability', (probability_position_x, probability_position_y),
                            font, fontsize,
                            (255, 255, 255), font_scale, cv2.LINE_AA)
                pretext = classnamelist[z]
                cv2.putText(apply_color_map_image, pretext, (samplename_position_x + x_move, samplename_position_y),
                            font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                if pre[y] > 0.9:  # 確率が90%を超える時
                    cv2.putText(apply_color_map_image, '{}%'.format(round(pre[z] * 100)),
                                (probability_position_x + x_move, probability_position_y), font, fontsize,
                                (0, 0, 255), font_scale, cv2.LINE_AA)

                else:
                    cv2.putText(apply_color_map_image, '{}%'.format(round(pre[z] * 100)),
                                (probability_position_x + x_move, probability_position_y), font, fontsize,
                                (255, 255, 255), font_scale, cv2.LINE_AA)

            cv2.putText(apply_color_map_image,
                        self.colormap_table[self.colormap_table_count % len(self.colormap_table)][0],
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)

            cv2.imshow("Please push Q button when you want to close the window.",
                       cv2.resize(apply_color_map_image, (800, 800)))
            cv2.imshow('Predict tag imaging', imaging)

            if self.initsavecount == 0 and self.savecount == 0:
                pass
            elif self.initsavecount != self.savecount:
                if os.path.exists(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber)):
                    for existnumber in range(len(glob.glob(self.savepath + '/*.png'))):
                        self.existnumber = existnumber + 1
                    print('同じファイルが存在しているので、ファイルを新規作成します')
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       apply_color_map_image)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
                else:
                    self.im_jp.imwrite(self.savepath + '/{:0>6}.png'.format(self.initsavecount + self.existnumber),
                                       apply_color_map_image)
                    self.initsavecount += 1
                    print('saveimage:{:0>6}'.format(self.initsavecount + self.existnumber))
            elif self.initsavecount == self.savecount:
                self.initsavecount = self.savecount = 0
                self.existnumber = 0
                print('Initialize savecount')

            if self.video_saveflag == True:
                self.out.write(cv2.resize(apply_color_map_image, (self.width, self.height)))
                self.out2.write(imaging)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                if self.video_saveflag == True:
                    self.out.release()
                    self.out2.release()
                    self.video_saveflag = False
                    print('録画終了')
                cv2.destroyAllWindows()
                print('Complete Cancel')
                break

            elif k == ord('n'):  # N
                self.colormap_table_count = self.colormap_table_count + 1

            # 処理後の時刻
            #t2 = time.time()

            # # 経過時間を表示
            # try:
            #     freq = 1 / (t2 - t1)
            #     print(f"フレームレート：{freq}fps")
            # except ZeroDivisionError:
            #     pass

        self.cam_manager.stop_acquisition()
        print('Stopped Camera')

    def save(self, savecount, savepath):
        self.savecount = savecount
        self.savepath = savepath

    def video_save(self, savepath, fps, height, width, color):
        self.video_savepath = savepath
        self.fps = fps
        self.height = height
        self.width = width
        tstr = dt.now().strftime('%Y_%m_%d_%H_%M_%S')
        self.result = self.video_savepath + '/{}.mp4'.format(tstr)
        self.result2 = self.video_savepath + '/tag_imaging_{}.mp4'.format(tstr)
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.result, self.fourcc, self.fps, (self.width, self.height), isColor=color)
        self.out2 = cv2.VideoWriter(self.result2, self.fourcc, self.fps, (self.background_width, self.background_height), isColor=color)
        self.video_saveflag = True
        print('録画開始')

    def video_finish(self):
        self.out.release()
        self.out2.release()
        self.video_saveflag = False
        print('録画終了')

    def min_max_normalization(self, frame):
        frame = frame.astype(int)
        vmin = frame.min()
        vmax = frame.max()
        frame = (frame - vmin).astype(float) / (vmax - vmin).astype(float)
        frame = frame * 255
        frame = frame.astype('uint8')
        return frame

    def min_max_flag(self):
        if self.norm == False:
            self.norm = True
        else:
            self.norm = False

    def detect_ellipse(self, numbeams, minsize, maxsize, binthresh):
        self.numbeams = numbeams
        self.minsize = minsize
        self.maxsize = maxsize
        self.binthresh = binthresh
        self.detectflag = 1