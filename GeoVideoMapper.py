import os
import csv
import math
import threading
import json
import yaml
from collections import defaultdict

import cv2
import numpy as np
import PySimpleGUI as sg

try:
    from pyproj import CRS, Transformer
except Exception:
    Transformer = None

try:
    import pandas as pd
except Exception:
    pd = None


# -------------------------
# Utility functions
# -------------------------

def lonlat_to_utm_local(lonlat_list):
    if Transformer is None:
        raise RuntimeError('pyproj 未安装，无法转换经纬度')
    first_lon, first_lat = lonlat_list[0]
    zone = int((first_lon + 180) / 6) + 1
    is_north = first_lat >= 0
    epsg = 32600 + zone if is_north else 32700 + zone
    utm_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    arr = []
    for lon, lat in lonlat_list:
        x, y = transformer.transform(lon, lat)
        arr.append([x, y])
    arr = np.array(arr, dtype=np.float64)
    origin = arr[0].copy()
    arr_local = arr - origin
    crs_info = {'type': 'utm', 'epsg': epsg, 'zone': zone, 'origin_utm': origin.tolist()}
    return arr_local, crs_info


def read_gcp_csv(path):
    rows = []
    with open(path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if len(rows) == 0:
        raise ValueError('GCP CSV 为空')
    cols = rows[0].keys()
    image_pts = []
    lonlat_list = []
    ground_m = []
    for r in rows:
        ix = float(r.get('image_x') or r.get('u'))
        iy = float(r.get('image_y') or r.get('v'))
        image_pts.append([ix, iy])
        if 'lon' in cols and 'lat' in cols:
            lon = float(r.get('lon'))
            lat = float(r.get('lat'))
            lonlat_list.append((lon, lat))
        elif 'longitude' in cols and 'latitude' in cols:
            lon = float(r.get('longitude'))
            lat = float(r.get('latitude'))
            lonlat_list.append((lon, lat))
        elif 'ground_x' in cols and 'ground_y' in cols:
            gx = float(r.get('ground_x')); gy = float(r.get('ground_y'))
            ground_m.append([gx, gy])
        else:
            raise ValueError('GCP CSV 必须包含 image_x,image_y 与 (lon,lat) 或 (ground_x,ground_y)')
    image_pts = np.array(image_pts, dtype=np.float32)
    if len(lonlat_list) > 0:
        gnd_local, crs_info = lonlat_to_utm_local(lonlat_list)
        return image_pts, gnd_local, crs_info
    else:
        return image_pts, np.array(ground_m, dtype=np.float64), {'type': 'local_m'}


def save_gcp_csv(path, img_pts, lonlat_list=None, ground_pts=None):
    # if lonlat_list given, write lon,lat; else ground_pts in meters
    with open(path, 'w', newline='', encoding='utf-8') as f:
        if lonlat_list is not None:
            writer = csv.DictWriter(f, fieldnames=['image_x','image_y','lon','lat'])
            writer.writeheader()
            for (x,y),(lon,lat) in zip(img_pts, lonlat_list):
                writer.writerow({'image_x': x, 'image_y': y, 'lon': lon, 'lat': lat})
        else:
            writer = csv.DictWriter(f, fieldnames=['image_x','image_y','ground_x','ground_y'])
            writer.writeheader()
            for (x,y),(gx,gy) in zip(img_pts, ground_pts):
                writer.writerow({'image_x': x, 'image_y': y, 'ground_x': gx, 'ground_y': gy})


def undistort_points(points, camera_matrix, dist_coeffs):
    """对图像点进行畸变校正"""
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    undistorted_points = cv2.undistortPoints(points, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted_points.reshape(-1, 2)


def compute_homography(image_pts, ground_pts, camera_matrix=None, dist_coeffs=None, ransac_thresh=3.0):
    if image_pts.shape[0] < 4:
        raise ValueError('至少需要 4 个 GCP 点')
    
    # 如果提供了相机参数，对图像点进行畸变校正
    if camera_matrix is not None and dist_coeffs is not None:
        image_pts = undistort_points(image_pts, camera_matrix, dist_coeffs)
        print("已对图像点进行畸变校正")
    
    H, mask = cv2.findHomography(image_pts, ground_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    mask = mask.reshape(-1).astype(bool)
    img_in = image_pts[mask]
    gnd_in = ground_pts[mask]
    if img_in.shape[0] >= 1:
        proj = cv2.perspectiveTransform(img_in.reshape(-1,1,2), H).reshape(-1,2)
        errs = np.linalg.norm(proj - gnd_in, axis=1)
        rmse = float(np.sqrt(np.mean(errs**2)))
    else:
        rmse = float('nan')
    # 打印单应矩阵
    print("Homography Matrix:")
    print(H)

    return H, mask, rmse


def bbox_to_corners(box):
    x,y,w,h = box
    return np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]], dtype=np.float32)


def box_to_ground_centroid(box, H, undistort=False, cam_mtx=None, dist=None):
    corners = bbox_to_corners(box).reshape(-1,1,2)
    if undistort and cam_mtx is not None and dist is not None:
        und = cv2.undistortPoints(corners, cam_mtx, dist, P=cam_mtx)
        corners_pix = und
    else:
        corners_pix = corners
    corners_g = cv2.perspectiveTransform(corners_pix, H).reshape(-1,2)
    return corners_g.mean(axis=0)


def frame_to_bytes(frame, scale=1.0):
    # convert BGR to PNG bytes for PySimpleGUI Image element
    if scale != 1.0:
        h,w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
    ret, buf = cv2.imencode('.png', frame)
    if not ret:
        return None
    return buf.tobytes()


# -------------------------
# GUI Layout and Handlers
# -------------------------

def build_window():
    sg.theme('SystemDefault')
    sg.set_options(element_padding=(4,4))

    left_col = [
        [sg.Text('1) Video', font=('Any',10,'bold'))],
        [sg.Input(key='-VIDEO_PATH-', enable_events=True), sg.FileBrowse('Load Video')],
        [sg.Text('Jump to frame:'), sg.Input('0', size=(8,1), key='-FRAME_INPUT-'), sg.Button('Go')],
        [sg.Slider(range=(0,100), orientation='h', size=(40,12), key='-FRAME_SLIDER-', enable_events=True)],
        [sg.Text('Frame size:'), sg.Text('W x H', key='-FRAME_SIZE-')],
        [sg.HorizontalSeparator()],
        [sg.Text('2) GCP (click on image to add)'), sg.Button('Import GCP CSV'), sg.Button('Export GCP CSV')],
        [sg.Text('Required: image_x,image_y,lon,lat  OR image_x,image_y,ground_x,ground_y')],
        [sg.Listbox(values=[], size=(40,8), key='-GCP_LIST-')],
        [sg.Button('Clear GCP')],
        [sg.Text('CSV Preview (format will be importable by the tool):')],
        [sg.Multiline('', size=(45,8), key='-GCP_CSV_PREVIEW-', disabled=True)],
        [sg.Button('Save GCP CSV')],
        [sg.HorizontalSeparator()],
        [sg.Text('3) Camera Parameters (optional)')],
        [sg.Input(key='-CAMERA_PATH-', enable_events=True), sg.FileBrowse('Load Camera Params')],
        [sg.Text('', key='-CAMERA_STATUS-')],
        [sg.HorizontalSeparator()],
        [sg.Text('4) Compute Homography')],
        [sg.Text('RANSAC thresh (px)'), sg.Input('3.0', size=(6,1), key='-RANSAC-')],
        [sg.Checkbox('Apply distortion correction', default=False, key='-UNDISTORT-')],
        [sg.Button('Compute H'), sg.Text('', key='-H_RMSE-')],
        [sg.HorizontalSeparator()],
        [sg.Text('5) Detections (optional)')],
        [sg.Input(key='-DETS_PATH-', enable_events=True), sg.FileBrowse('Load Detections CSV')],
        [sg.Button('Map Detections'), sg.Button('Export Results')],
        [sg.Checkbox('Visualize overlay video', key='-VISUAL-', default=True)],
        [sg.ProgressBar(max_value=100, orientation='h', size=(40,12), key='-PROG-')]
    ]

    right_col = [
        [sg.Text('Video Frame', font=('Any', 12, 'bold'))],
        [sg.Graph(canvas_size=(800,600), graph_bottom_left=(0,0), graph_top_right=(800,600), key='-GRAPH-', enable_events=True, drag_submits=False)],
        [sg.Text('Click on the image to add GCP point. Then input lon,lat in the popup.')],
        [sg.Button('Prev Frame'), sg.Button('Next Frame'), sg.Button('Play/Pause')]
    ]

    layout = [[sg.Column(left_col), sg.VerticalSeparator(), sg.Column(right_col)]]
    return sg.Window('LatLon→UTM→Video Measurement (GUI)', layout, finalize=True, resizable=True)


def update_gcp_csv_preview(window, img_pts, lonlat_list, ground_pts):
    """Generate CSV preview text and update the Multiline element."""
    if len(img_pts) == 0:
        window['-GCP_CSV_PREVIEW-'].update('')
        return
    lines = []
    if lonlat_list is not None and len(lonlat_list) == len(img_pts):
        lines.append('image_x,image_y,lon,lat')
        for (x,y),(lon,lat) in zip(img_pts, lonlat_list):
            lines.append(f'{int(x)},{int(y)},{lon:.12f},{lat:.12f}')
    elif ground_pts is not None and len(ground_pts) == len(img_pts):
        lines.append('image_x,image_y,ground_x,ground_y')
        for (x,y),(gx,gy) in zip(img_pts, ground_pts.tolist()):
            lines.append(f'{int(x)},{int(y)},{gx:.6f},{gy:.6f}')
    else:
        # no geo info available yet - only image pixels
        lines.append('image_x,image_y')
        for (x,y) in img_pts:
            lines.append(f'{int(x)},{int(y)}')
    txt = '\n'.join(lines)
    window['-GCP_CSV_PREVIEW-'].update(txt)


def run_gui():
    window = build_window()

    video_path = None
    cap = None
    total_frames = 0
    curr_frame_idx = 0
    frame_img = None
    frame_scale = 1.0
    playing = False

    gcp_img_pts = []      # list of (x,y) pixels
    gcp_lonlat = []       # list of (lon,lat) if provided
    gcp_ground = None     # np.array ground coords in meters after conversion
    crs_info = None

    H = None
    H_rmse = None

    detections = []
    mapped_records = []
    
    # 相机参数
    camera_matrix = None
    dist_coeffs = None

    def open_video(path):
        nonlocal cap, total_frames, curr_frame_idx
        if cap is not None:
            cap.release()
            cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sg.popup_error('无法打开视频: ' + path)
            return False
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        curr_frame_idx = 0
        window['-FRAME_SLIDER-'].update(range=(0, max(0,total_frames-1)))
        return True

    def read_frame(idx):
        nonlocal cap
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, fr = cap.read()
        if not ret:
            return None
        return fr

    def update_graph_with_frame(fr):
        nonlocal frame_scale
        g = window['-GRAPH-']
        gw, gh = g.get_size()
        h, w = fr.shape[:2]
        sx = gw / w; sy = gh / h
        frame_scale = min(sx, sy)
        display = cv2.resize(fr, (int(w*frame_scale), int(h*frame_scale)))
        imgbytes = frame_to_bytes(display)
        g.erase()
        g.draw_image(data=imgbytes, location=(0, gh))
        for i, (x,y) in enumerate(gcp_img_pts):
            gx = int(x*frame_scale); gy = int(y*frame_scale)
            g.draw_circle((gx, gh-gy), 5, fill_color='green')
            g.draw_text(str(i+1), (gx+8, gh-gy+8))

    def load_camera_params(path):
        """加载相机参数（JSON或YAML格式）"""
        nonlocal camera_matrix, dist_coeffs
        try:
            if path.lower().endswith('.json'):
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                with open(path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
            
            # 提取相机矩阵和畸变系数
            if 'camera_matrix' in data:
                camera_matrix = np.array(data['camera_matrix'], dtype=np.float64)
            elif 'K' in data:
                camera_matrix = np.array(data['K'], dtype=np.float64)
            
            if 'dist_coeffs' in data:
                dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float64)
            elif 'dist' in data:
                dist_coeffs = np.array(data['dist'], dtype=np.float64)
            
            if camera_matrix is None or dist_coeffs is None:
                sg.popup_error('相机参数文件中缺少必要的字段')
                return False
                
            window['-CAMERA_STATUS-'].update(f'已加载相机参数: K shape={camera_matrix.shape}, dist len={dist_coeffs.size}')
            return True
            
        except Exception as e:
            sg.popup_error(f'加载相机参数失败: {str(e)}')
            return False

    while True:
        event, values = window.read(timeout=50)
        if event == sg.WIN_CLOSED:
            break

        if event == '-VIDEO_PATH-':
            video_path = values['-VIDEO_PATH-']
            if video_path and os.path.exists(video_path):
                ok = open_video(video_path)
                if ok:
                    fr = read_frame(0)
                    if fr is not None:
                        frame_img = fr
                        h,w = fr.shape[:2]
                        window['-FRAME_SIZE-'].update(f'{w} x {h}')
                        update_graph_with_frame(fr)
                        window['-FRAME_SLIDER-'].update(value=0)

        if event == '-CAMERA_PATH-':
            camera_path = values['-CAMERA_PATH-']
            if camera_path and os.path.exists(camera_path):
                load_camera_params(camera_path)

        if event == 'Go':
            try:
                idx = int(values['-FRAME_INPUT-'])
            except Exception:
                idx = 0
            if cap is not None and 0 <= idx < total_frames:
                curr_frame_idx = idx
                fr = read_frame(curr_frame_idx)
                if fr is not None:
                    frame_img = fr
                    update_graph_with_frame(fr)
                    window['-FRAME_SLIDER-'].update(value=curr_frame_idx)

        if event == '-FRAME_SLIDER-':
            val = int(values['-FRAME_SLIDER-'])
            if cap is not None and val != curr_frame_idx:
                curr_frame_idx = val
                fr = read_frame(curr_frame_idx)
                if fr is not None:
                    frame_img = fr
                    update_graph_with_frame(fr)
                    window['-FRAME_INPUT-'].update(curr_frame_idx)

        if event == 'Prev Frame':
            if cap is not None:
                curr_frame_idx = max(0, curr_frame_idx-1)
                fr = read_frame(curr_frame_idx)
                if fr is not None:
                    frame_img = fr
                    update_graph_with_frame(fr)
                    window['-FRAME_SLIDER-'].update(curr_frame_idx); window['-FRAME_INPUT-'].update(curr_frame_idx)

        if event == 'Next Frame':
            if cap is not None:
                curr_frame_idx = min(total_frames-1, curr_frame_idx+1)
                fr = read_frame(curr_frame_idx)
                if fr is not None:
                    frame_img = fr
                    update_graph_with_frame(fr)
                    window['-FRAME_SLIDER-'].update(curr_frame_idx); window['-FRAME_INPUT-'].update(curr_frame_idx)

        if event == 'Play/Pause':
            playing = not playing

        if playing and cap is not None:
            curr_frame_idx += 1
            if curr_frame_idx >= total_frames:
                curr_frame_idx = 0
            fr = read_frame(curr_frame_idx)
            if fr is not None:
                frame_img = fr
                update_graph_with_frame(fr)
                window['-FRAME_SLIDER-'].update(curr_frame_idx); window['-FRAME_INPUT-'].update(curr_frame_idx)

        if event == '-GRAPH-':
            if frame_img is None:
                continue
            x, y = values['-GRAPH-']
            gw, gh = window['-GRAPH-'].get_size()
            img_x = int(x / frame_scale)
            img_y = int((gh - y) / frame_scale)
            form = [[sg.Text('Pixel:'), sg.Text(f'({img_x},{img_y})')],
                    [sg.Text('输入 lon lat (空格分隔)，如: 113.1041225 23.4611129')],
                    [sg.Input(key='-LONLAT-')],
                    [sg.Button('OK'), sg.Button('Cancel')]]
            pop = sg.Window('GCP Input', form)
            evt, vals = pop.read()
            if evt == 'OK':
                txt = vals['-LONLAT-'].strip()
                if txt == '':
                    sg.popup('未输入 lon/lat，跳过。')
                else:
                    parts = txt.split()
                    if len(parts) == 2:
                        try:
                            lon = float(parts[0]); lat = float(parts[1])
                            gcp_img_pts.append((img_x, img_y))
                            gcp_lonlat.append((lon, lat))
                            # update preview
                            update_gcp_csv_preview(window, gcp_img_pts, gcp_lonlat, gcp_ground)
                        except Exception:
                            sg.popup_error('经纬度解析失败')
                    else:
                        sg.popup_error('请输入两个浮点数：lon lat')
            pop.close()
            lb = [f'{i+1}: px=({x},{y}) lonlat={gcp_lonlat[i] if i < len(gcp_lonlat) else ""}' for i,(x,y) in enumerate(gcp_img_pts)]
            window['-GCP_LIST-'].update(lb)
            update_graph_with_frame(frame_img)

        if event == 'Import GCP CSV':
            path = sg.popup_get_file('Select GCP CSV', file_types=(('CSV Files','*.csv'),))
            if path:
                try:
                    img_pts, gnd_pts, crs = read_gcp_csv(path)
                    gcp_img_pts = [(int(x), int(y)) for x,y in img_pts.tolist()]
                    if crs.get('type') == 'utm' or crs.get('type') == 'local_m':
                        gcp_ground = gnd_pts
                        gcp_lonlat = []
                    crs_info = crs
                    window['-GCP_LIST-'].update([f'{i+1}: px=({x},{y})' for i,(x,y) in enumerate(gcp_img_pts)])
                    update_graph_with_frame(frame_img)
                    update_gcp_csv_preview(window, gcp_img_pts, gcp_lonlat if len(gcp_lonlat)==len(gcp_img_pts) else None, gcp_ground)
                    sg.popup('Imported GCP OK', f'{len(gcp_img_pts)} points loaded')
                except Exception as e:
                    sg.popup_error('Import GCP failed', str(e))

        if event == 'Export GCP CSV':
            path = sg.popup_get_file('Save GCP CSV', save_as=True, file_types=(('CSV Files','*.csv'),), default_extension='csv')
            if path:
                try:
                    # if we have lonlat list, save lonlat, else save ground
                    if len(gcp_lonlat) == len(gcp_img_pts) and len(gcp_lonlat)>0:
                        save_gcp_csv(path, gcp_img_pts, lonlat_list=gcp_lonlat)
                    elif 'gcp_ground' in locals() and gcp_ground is not None:
                        save_gcp_csv(path, gcp_img_pts, ground_pts=gcp_ground)
                    else:
                        sg.popup_error('缺少地理坐标信息，无法导出')
                        continue
                    sg.popup('Saved')
                except Exception as e:
                    sg.popup_error('Save failed', str(e))

        if event == 'Save GCP CSV':
            # Save the CSV preview content directly to a file chosen by user
            txt = window['-GCP_CSV_PREVIEW-'].get()
            if txt.strip() == '':
                sg.popup_error('CSV 预览为空，无法保存')
            else:
                path = sg.popup_get_file('Save GCP CSV', save_as=True, file_types=(('CSV Files','*.csv'),), default_extension='csv')
                if path:
                    try:
                        with open(path, 'w', encoding='utf-8') as f:
                            f.write(txt)
                        sg.popup('Saved CSV preview to', path)
                    except Exception as e:
                        sg.popup_error('Save failed', str(e))

        if event == 'Clear GCP':
            gcp_img_pts = []
            gcp_lonlat = []
            gcp_ground = None
            crs_info = None
            window['-GCP_LIST-'].update([])
            update_graph_with_frame(frame_img)
            update_gcp_csv_preview(window, gcp_img_pts, gcp_lonlat, gcp_ground)

        if event == 'Compute H':
            if len(gcp_img_pts) < 4:
                sg.popup_error('至少需要 4 个 GCP')
            else:
                try:
                    img_pts_arr = np.array(gcp_img_pts, dtype=np.float32)
                    if len(gcp_lonlat) == len(gcp_img_pts) and len(gcp_lonlat)>0:
                        ground_pts_local, crs = lonlat_to_utm_local(gcp_lonlat)
                        crs_info = crs
                        gcp_ground = ground_pts_local
                        # update preview to show ground_x,ground_y too
                        update_gcp_csv_preview(window, gcp_img_pts, None, gcp_ground)
                        window['-GCP_LIST-'].update([f'{i+1}: px=({x},{y})' for i,(x,y) in enumerate(gcp_img_pts)])
                    elif 'gcp_ground' in locals() and gcp_ground is not None and len(gcp_ground)==len(gcp_img_pts):
                        ground_pts_local = gcp_ground
                    else:
                        sg.popup_error('缺少地理坐标或 ground 坐标。请导入带 lon/lat 的 GCP CSV 或使用交互输入。')
                        continue
                    
                    # 根据用户选择决定是否应用畸变校正
                    apply_undistort = values['-UNDISTORT-'] and camera_matrix is not None and dist_coeffs is not None
                    H, mask, rmse = compute_homography(
                        img_pts_arr, 
                        ground_pts_local, 
                        camera_matrix if apply_undistort else None,
                        dist_coeffs if apply_undistort else None,
                        ransac_thresh=float(values['-RANSAC-'])
                    )
                    H_rmse = rmse
                    window['-H_RMSE-'].update(f'RMSE: {rmse:.3f} m')
                    sg.popup('Compute H done', f'RMSE: {rmse:.3f} m')
                except Exception as e:
                    sg.popup_error('Compute H failed', str(e))

        if event == '-DETS_PATH-':
            det_path = values['-DETS_PATH-']
            if det_path and os.path.exists(det_path):
                try:
                    if pd is not None:
                        df = pd.read_csv(det_path)
                        if 'id' not in df.columns:
                            df['id'] = 0
                        detections = df.to_dict('records')
                    else:
                        detections = []
                        with open(det_path, 'r', newline='', encoding='utf-8') as f:
                            r = csv.DictReader(f)
                            for it in r:
                                detections.append({'frame': int(it['frame']), 'id': it.get('id',0), 'x': float(it['x']), 'y': float(it['y']), 'w': float(it['w']), 'h': float(it['h'])})
                    sg.popup('Detections loaded', f'{len(detections)} rows')
                except Exception as e:
                    sg.popup_error('Load detections failed', str(e))

        if event == 'Map Detections':
            if 'H' not in locals() and H is None:
                sg.popup_error('先计算 H (Compute H)')
            elif len(detections) == 0:
                sg.popup_error('先加载 detections CSV')
            else:
                mapped_records = []
                for row in detections:
                    box = (float(row['x']), float(row['y']), float(row['w']), float(row['h']))
                    centroid_ground = box_to_ground_centroid(box, H)
                    px = float(row['x']) + float(row['w'])/2.0
                    py = float(row['y']) + float(row['h'])/2.0
                    mapped_records.append({'frame': int(row['frame']), 'id': row.get('id',0), 'px': px, 'py': py, 'gx': float(centroid_ground[0]), 'gy': float(centroid_ground[1])})
                sg.popup('Mapping done', f'Mapped {len(mapped_records)} records')

        if event == 'Export Results':
            out_dir = sg.popup_get_folder('Select output folder', default_path='out')
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
                if H is not None:
                    np.save(os.path.join(out_dir, 'homography.npy'), H)
                if len(mapped_records) > 0:
                    with open(os.path.join(out_dir, 'points_ground.csv'), 'w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=['frame','id','px','py','gx','gy'])
                        w.writeheader(); w.writerows(mapped_records)
                if len(mapped_records) > 0:
                    dist_records = []
                    grp = defaultdict(list)
                    for r in mapped_records:
                        grp[r['id']].append(r)
                    for oid, lst in grp.items():
                        lst_sorted = sorted(lst, key=lambda x: x['frame'])
                        for i in range(1, len(lst_sorted)):
                            a = lst_sorted[i-1]; b = lst_sorted[i]
                            d = math.hypot(b['gx']-a['gx'], b['gy']-a['gy'])
                            dist_records.append({'id': oid, 'frame_from': a['frame'], 'frame_to': b['frame'], 'gx_from': a['gx'], 'gy_from': a['gy'], 'gx_to': b['gx'], 'gy_to': b['gy'], 'dist_m': d})
                    with open(os.path.join(out_dir, 'distances.csv'), 'w', newline='', encoding='utf-8') as f:
                        w = csv.DictWriter(f, fieldnames=['id','frame_from','frame_to','gx_from','gy_from','gx_to','gy_to','dist_m'])
                        w.writeheader(); w.writerows(dist_records)
                if values['-VISUAL-'] and 'video_path' in locals() and os.path.exists(values['-VIDEO_PATH-']):
                    try:
                        out_video_path = os.path.join(out_dir, 'result_overlay.mp4')
                        cap2 = cv2.VideoCapture(values['-VIDEO_PATH-'])
                        fps = cap2.get(cv2.CAP_PROP_FPS) or 25
                        w = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w,h))
                        rec_index = defaultdict(list)
                        for r in mapped_records:
                            rec_index[int(r['frame'])].append(r)
                        frame_id = 0
                        trajs = defaultdict(list)
                        while True:
                            ret, fr = cap2.read()
                            if not ret: break
                            if frame_id in rec_index:
                                for r in rec_index[frame_id]:
                                    cx = int(round(r['px'])); cy = int(round(r['py']))
                                    cv2.circle(fr, (cx,cy), 4, (0,255,0), -1)
                                    cv2.putText(fr, f"id:{r['id']} ({r['gx']:.2f}m,{r['gy']:.2f}m)", (cx+6, cy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                                    trajs[r['id']].append((cx,cy))
                            for oid, pts in trajs.items():
                                for i in range(1, len(pts)):
                                    cv2.line(fr, pts[i-1], pts[i], (255,0,0), 2)
                            writer.write(fr)
                            frame_id += 1
                        cap2.release(); writer.release()
                        sg.popup('Exported overlay video', out_video_path)
                    except Exception as e:
                        sg.popup_error('Failed export overlay', str(e))
                sg.popup('Export finished', f'Files saved to {out_dir}')

    window.close()


if __name__ == '__main__':
    run_gui()