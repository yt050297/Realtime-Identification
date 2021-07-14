import cv2

# 動画を読み込み、FPSを変更して別名で保存する関数
def m_speed_change(path_in, path_out, fps_new, color_flag):
    # 動画読み込みの設定
    movie = cv2.VideoCapture(path_in)

    # 動画ファイル保存用の設定
    fps = int(movie.get(cv2.CAP_PROP_FPS))                                  # 元動画のFPSを取得
    print(fps)
    #fps_new = int(fps * scale_factor)                                       # 動画保存時のFPSはスケールファクターをかける
    w = int(movie.get(cv2.CAP_PROP_FRAME_WIDTH))                            # 動画の横幅を取得
    h = int(movie.get(cv2.CAP_PROP_FRAME_HEIGHT))                           # 動画の縦幅を取得
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')                     # 動画保存時のfourcc設定（mp4用）
    video = cv2.VideoWriter(path_out, fourcc, fps_new, (w, h), color_flag)  # 動画の仕様（ファイル名、fourcc, FPS, サイズ）

    # ファイルからフレームを1枚ずつ取得して動画処理後に保存する
    while True:
        ret, frame = movie.read()        # フレームを取得
        video.write(frame)
        #print(frame)# 動画を保存する
        # フレームが取得できない場合はループを抜ける
        if not ret:
            break
    # 撮影用オブジェクトとウィンドウの解放
    movie.release()
    return

path = 'C:/Users/yt050/Desktop/imagingcv2/ok_video/'         # 元動画のパス
file_name = '2021_07_14_18_58_02_card3.mp4'
path_in = path + file_name
#scale_factor = 0.1              # FPSにかけるスケールファクター
new_fps = 1.35
color_flag = True               # カラー動画はTrue, グレースケール動画はFalse
path_out = path + '2021_07_14_18_58_02_card3_changed_{}fps.mp4'.format(new_fps)      # 保存する動画のパス

#print(path_in)
#print(path_out)
# 動画の再生速度を変更する関数を実行
m_speed_change(path_in, path_out, new_fps, color_flag)