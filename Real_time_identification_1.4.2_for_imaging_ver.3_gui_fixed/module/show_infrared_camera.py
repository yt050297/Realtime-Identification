from module.camera_manager import CameraManager
from module.camera_manager import TriggerType
from module.camera_manager import AcquisitionMode
from module.camera_manager import AutoExposureMode
from module.camera_manager import AutoGainMode
from module.imread_imwrite_japanese import ImreadImwriteJapanese
from module.create_reference import CreateReference
from module.fwhm import FWHM
from module.autopolarizer import AutoPolarizer
import cv2
import time
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
import os
from datetime import datetime as dt
import glob

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

        self.background_width = 1440
        self.background_height = 1440

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
            '''
            cv2.putText(apply_color_map_image,
                        self.colormap_table[self.colormap_table_count % len(self.colormap_table)][0],
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            '''
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
                y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル

                cv2.putText(apply_color_map_image, 'Predict sample', (samplename_position_x, samplename_position_y),
                            font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)
                cv2.putText(apply_color_map_image, 'Probability', (probability_position_x, probability_position_y),
                            font, fontsize,
                            (255, 255, 255), font_scale, cv2.LINE_AA)
                pretext = classnamelist[y]
                cv2.putText(apply_color_map_image, pretext, (samplename_position_x + x_move, samplename_position_y),
                            font, fontsize, (255, 255, 255), font_scale, cv2.LINE_AA)

                if pre[y] > 0.9:  # 確率が90%を超える時
                    cv2.putText(apply_color_map_image, '{}%'.format(round(pre[y] * 100)),
                                (probability_position_x + x_move, probability_position_y), font, fontsize,
                                (0, 0, 255), font_scale, cv2.LINE_AA)
                else:
                    cv2.putText(apply_color_map_image, '{}%'.format(round(pre[y] * 100)),
                                (probability_position_x + x_move, probability_position_y), font, fontsize,
                                (255, 255, 255), font_scale, cv2.LINE_AA)

            '''cv2.putText(apply_color_map_image,
                        self.colormap_table[self.colormap_table_count % len(self.colormap_table)][0],
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2)
            '''
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
            try:
                freq = 1 / (t2 - t1)
                print(f"フレームレート：{freq}fps")
            except ZeroDivisionError:
                pass

        self.cam_manager.stop_acquisition()
        print('Stopped Camera')

    '''
    def write_backgrand_imaging(self, height, width):
        img = np.zeros((height, width, 3), np.uint8)
        return img

    def write_onepixel_imaging(self, img, fromleft, fromupper, color, width):
        img = cv2.rectangle(img, (fromleft, fromupper), (fromleft + width, fromupper + width), color, -1)
        return img

    def color_append(self, pre_num, pre_acc, imaging, fromleft, fromupper, pixel_size):
        if pre_num == 4:
            imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255*pre_acc, 255*pre_acc, 255*pre_acc), pixel_size)
        elif pre_num == 0:
            imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255*pre_acc), pixel_size)
        elif pre_num == 1:
            imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255*pre_acc, 0), pixel_size)
        elif pre_num == 2:
            imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255*pre_acc, 0, 0), pixel_size)
        elif pre_num == 3:
            imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255*pre_acc, 255*pre_acc), pixel_size)

        return imaging

    def save_video(self):
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
            self.out.write(imaging)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            if self.video_saveflag == True:
                self.out.release()
                self.video_saveflag = False
                print('録画終了')
            cv2.destroyAllWindows()
            print('Complete Cancel')

        elif k == ord('n'):  # N
            self.colormap_table_count = self.colormap_table_count + 1

    '''

    def realtime_identification_color(self, classnamelist, model, trigger, gain, exp, im_size_width, im_size_height,
                                      flip,initial_height, initial_side, pixel_height, pixel_side, error):

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

        ############イメージング表示用
        fromleft = 10  # 最初のピクセルの左端からの位置
        fromupper = 10  # 最初のピクセルの上端からの位置
        #background_width = 1440
        #background_height = 1440
        pixel_size = 10  # 移動距離＆1pixelのサイズ

        #######ここからステージ####################################################################
        print('ok')
        port = 'COM3'
        #初期位置
        side_stage = initial_side
        height_stage = initial_height
        #side_stage = 50000    ###1um/pulse
        #side_stage = 78000  ##横8cm
        #side_stage = 66000   #横3列
        #height_stage = 0  ###1um/pulse

        #ピクセル数
        side_pixel = pixel_side
        height_pixel = pixel_height
        #side_pixel = 40   ###通常時横5cm
        #side_pixel = 58
        #side_pixel = 70   ###横8cm
        #side_pixel = 58 #横3列
        #height_pixel = 20
        #height_pixel = 40
        #height_pixel = 55

        #解像度設定
        side_resolution = 1000
        height_resolution = 1000
        side_length = side_resolution * side_pixel
        spd_min = 19000  # 最小速度[PPS]
        spd_max = 20000  # 最大速度[PPS]
        acceleration_time = 0 # 加減速時間[mS]
        sec = 0.8
        ref_time = 0.19

        #誤差修正
        yoko_sec = error
        ##yoko_sec = 0.35   #通常時
        #yoko_sec = 0.44  #サンプル長い時


        def write_backgrand_imaging(height, width):
            img = np.zeros((height, width, 3), np.uint8)
            return img

        def write_onepixel_imaging(img, fromleft, fromupper, color, width):
            img = cv2.rectangle(img, (fromleft, fromupper), (fromleft + width, fromupper + width), color, -1)
            return img

        def color_append(pre_num, pre_acc, imaging, fromleft, fromupper, pixel_size):
            if pre_num == 0:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 102), pixel_size)
            elif pre_num == 1:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 178), pixel_size)
            elif pre_num == 2:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255), pixel_size)
            elif pre_num == 3:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (102, 0, 0), pixel_size)
            elif pre_num == 4:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (178, 0, 0), pixel_size)
            elif pre_num == 5:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 0, 0), pixel_size)
            elif pre_num == 6:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 102, 102), pixel_size)
            elif pre_num == 7:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 178, 178), pixel_size)
            elif pre_num == 8:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255, 255), pixel_size)
            elif pre_num == 9:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 255, 255), pixel_size)
            elif pre_num == 10:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 178, 178), pixel_size)
            elif pre_num == 11:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 102, 102), pixel_size)
            elif pre_num == 12:
                imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 255, 255), pixel_size)


            # elif pre_num == 1:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255), pixel_size)
            # elif pre_num == 2:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255, 0), pixel_size)
            # elif pre_num == 3:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 0, 0), pixel_size)
            # elif pre_num == 4:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 255, 255), pixel_size)
            # elif pre_num == 5:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 255, 0), pixel_size)
            # elif pre_num == 6:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (255, 0, 255), pixel_size)
            # elif pre_num == 7:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (125, 125, 125), pixel_size)


            # elif pre_num == 1:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255), pixel_size)
            # elif pre_num == 2:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255*0.6), pixel_size)
            # elif pre_num == 3:
            #     imaging = write_onepixel_imaging(imaging, fromleft, fromupper, (0, 0, 255*0.2), pixel_size)

            return imaging

        def save_video(apply_color_map_image):
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
                self.out.write(imaging)

            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                if self.video_saveflag == True:
                    self.out.release()
                    self.video_saveflag = False
                    print('録画終了')
                cv2.destroyAllWindows()
                print('Complete Cancel')

            elif k == ord('n'):  # N
                self.colormap_table_count = self.colormap_table_count + 1

        ####イメージングの背景表示
        imaging = write_backgrand_imaging(self.background_height, self.background_width)
        cv2.imshow('image', imaging)

        #Y = np.zeros((height_pixel, side_pixel))

        polarizer = AutoPolarizer(port=port)

        print('setting_time')
        polarizer.set_speed(spd_min, spd_max, acceleration_time)
        time.sleep(2)

        print("Reset")
        polarizer.reset()
        time.sleep(25)
        print('初期位置設定')
        polarizer._set_position_relative(1, side_stage)
        time.sleep(30)
        polarizer._set_position_relative(2, height_stage)
        time.sleep(5)



        ##shoki
        polarizer._set_position_relative(2, height_resolution)
        time.sleep(sec)
        #i = j = 0
        n=0
        if trigger == "software":
            self.cam_manager.execute_software_trigger()

        frame = self.cam_manager.get_next_image()
        # if frame is None:
        #     continue

        if self.norm == True:
            frame = self.min_max_normalization(frame)

        # 読み込んだフレームを書き込み
        if flip == 'normal':
            pass
        elif flip == 'flip':
            frame = cv2.flip(frame, 1)  # 画像を左右反転

        resize_image = cv2.resize(frame, (im_size_width, im_size_height))
        # #     # 疑似カラーを付与
        # apply_color_map_image = cv2.applyColorMap(frame, self.colormap_table[self.colormap_table_count % len(self.colormap_table)][1])
        # cv2.imshow("Please push Q button when you want to close the window.",cv2.resize(apply_color_map_image, (800, 800)))

        X = []
        X.append(resize_image)
        X = np.array(X)
        X = X.astype("float") / 256

        X.resize(X.shape[0], X.shape[1], X.shape[2], 1)

        pre = model.predict(X)

        pre = pre[0]
        y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
        #Y[i][j] = y
        #print(i, j)
        #print(pre)
        color_append(y, pre[y], imaging, fromleft, fromupper, pixel_size)
        cv2.imshow('image', imaging)
        save_video(imaging)
        print(n)
        #print('予測結果:{}'.format(y))
        n=n+1

        for i in range(height_pixel):
            if i % 2 == 0:
                polarizer._set_position_relative(1, -side_length)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                time.sleep(yoko_sec)

                for j in range(1, side_pixel):
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

                    pre = model.predict(X)
                    pre = pre[0]

                    y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
                    #Y[i][j] = y
                    #print(i, j)
                    fromleft = fromleft + pixel_size
                    color_append(y, pre[y], imaging, fromleft, fromupper, pixel_size)
                    cv2.imshow('image', imaging)
                    save_video(imaging)
                    print(n)
                    #print('予測結果{}'.format(y))
                    n=n+1
                    t2 = time.time()
                    time1 = t2 - t1
                    time_chousetu = ref_time - time1
                    time.sleep(time_chousetu)
                    t3 = time.time()
                    fps = 1/(t3-t1)
                    print(fps)

            else:
                polarizer._set_position_relative(1, side_length)  # 引数一つ目、1:一軸、2:2軸、W:両軸
                time.sleep(yoko_sec)
                # time.sleep(sec)
                for j in range(1, side_pixel):
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

                    pre = model.predict(X)
                    pre = pre[0]
                    y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
                    #Y[i][side_pixel-1-j] = y
                    #print(i, side_pixel-1-j)
                    fromleft = fromleft - pixel_size
                    color_append(y, pre[y], imaging, fromleft, fromupper, pixel_size)
                    cv2.imshow('image', imaging)
                    save_video(imaging)
                    print(n)
                    #print('予測結果:{}'.format(y))
                    n=n+1
                    t2 = time.time()
                    time1 = t2 - t1
                    time_chousetu = ref_time - time1
                    time.sleep(time_chousetu)
                    t3 = time.time()
                    fps = 1/(t3-t1)
                    print(fps)

            time.sleep(sec)
            polarizer._set_position_relative(2, height_resolution)
            time.sleep(sec)

            fromupper = fromupper + pixel_size

            if i % 2 == 0:
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

                pre = model.predict(X)
                pre = pre[0]
                y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
                #Y[i+1][j] = y
                #print(i+1, j)
                #fromleft = fromleft + pixel_size
                color_append(y, pre[y], imaging, fromleft, fromupper, pixel_size)
                cv2.imshow('image', imaging)
                save_video(imaging)
                print(n)
                #print('予測結果:{}'.format(y))
                n=n+1


            else:
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

                pre = model.predict(X)
                pre = pre[0]
                y = pre.argmax()  # preがそれぞれの予測確率で一番高いものを取ってきている。Y_testはone-hotベクトル
                #Y[i+1][side_pixel-1-j] = y
                #print(i+1, side_pixel-1-j)
                #fromleft = fromleft + pixel_size
                color_append(y, pre[y], imaging, fromleft, fromupper, pixel_size)
                cv2.imshow('image', imaging)
                save_video(imaging)
                print(n)
                #print('予測結果:{}'.format(y))
                n=n+1


        time.sleep(1)
        self.video_finish()
        polarizer.stop()

        #cv2.imshow('image',imaging)
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
        self.fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        self.fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(self.result, self.fourcc, self.fps, (self.background_width, self.background_height), isColor=color)
        self.video_saveflag = True
        print('録画開始')

    def video_finish(self):
        self.video_saveflag = False
        self.out.release()
        cv2.destroyAllWindows()
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
