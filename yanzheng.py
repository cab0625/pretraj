# homography_validator_with_pred_lonlat_and_export.py
import os
import io
import csv
import math
import re
import cv2
import numpy as np
import PySimpleGUI as sg

# optional
try:
    from pyproj import CRS, Transformer
except Exception:
    Transformer = None

try:
    import pandas as pd
except Exception:
    pd = None

# ---------- helpers ----------

def parse_H_text(s: str):
    toks = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    if len(toks) != 9:
        return None
    arr = np.array([float(t) for t in toks], dtype=np.float64).reshape((3,3))
    if abs(arr[2,2]) > 1e-12:
        arr = arr / arr[2,2]
    return arr

def load_H_npy(path):
    try:
        arr = np.load(path)
        if arr.shape == (3,3):
            H = arr.astype(np.float64)
            if abs(H[2,2])>1e-12:
                H = H / H[2,2]
            return H
    except Exception as e:
        print("load H error:", e)
    return None

def image_to_ground(pt_xy, H):
    arr = np.array([[[float(pt_xy[0]), float(pt_xy[1])]]], dtype=np.float64)
    proj = cv2.perspectiveTransform(arr, H).reshape(2)
    return float(proj[0]), float(proj[1])

def lonlat_to_utm_local_single(lon, lat, origin_lonlat):
    """
    Convert lon/lat -> local UTM (meters) relative to origin_lonlat.
    Returns (x_local, y_local), origin_info where origin_info = (ox, oy, epsg)
    """
    if Transformer is None:
        raise RuntimeError("pyproj 未安装，无法进行 lon/lat <-> UTM 转换。")
    first_lon, first_lat = origin_lonlat
    zone = int((first_lon + 180) / 6) + 1
    is_north = first_lat >= 0
    epsg = 32600 + zone if is_north else 32700 + zone
    utm_crs = CRS.from_epsg(epsg)
    transformer = Transformer.from_crs('EPSG:4326', utm_crs, always_xy=True)
    x, y = transformer.transform(lon, lat)
    ox, oy = transformer.transform(first_lon, first_lat)
    return (x - ox, y - oy), (ox, oy, epsg)

def utm_local_to_lonlat(gx, gy, origin_info):
    """
    Given local gx,gy and origin_info=(ox,oy,epsg), return (lon, lat).
    """
    if Transformer is None:
        raise RuntimeError("pyproj 未安装，无法进行 lon/lat <-> UTM 转换。")
    ox, oy, epsg = origin_info
    utm_crs = CRS.from_epsg(int(epsg))
    transformer = Transformer.from_crs(utm_crs, 'EPSG:4326', always_xy=True)
    world_x = gx + ox
    world_y = gy + oy
    lon, lat = transformer.transform(world_x, world_y)
    return lon, lat

# ---------- GUI ----------

def build_window():
    sg.theme('SystemDefault')

    left_col = [
        [sg.Text('Video:'), sg.Input(key='-VIDEO-', enable_events=True, size=(40,1)), sg.FileBrowse('Load')],
        [sg.Text('Frame index:'), sg.Input('0', key='-FRAME-', size=(8,1)), sg.Button('Go')],
        [sg.Slider(range=(0,100), orientation='h', size=(40,12), key='-SLIDER-', enable_events=True)],
        [sg.Text('H (paste matrix like your example, or Load .npy):')],
        [sg.Multiline('', size=(40,6), key='-H_TEXT-')],
        [sg.Button('Load H .npy'), sg.Button('Apply H'), sg.Text('Status:', size=(6,1)), sg.Text('', key='-H_STATUS-')],
        [sg.HorizontalSeparator()],
        [sg.Text('Origin lon lat (used when H was computed):')],
        [sg.Text('Origin lon:'), sg.Input('', size=(12,1), key='-ORIGIN_LON-'), sg.Text('lat:'), sg.Input('', size=(12,1), key='-ORIGIN_LAT-')],
        [sg.Text('（若不填写，程序将自动把你输入的第一个 lon/lat 设为 origin）')],
        [sg.HorizontalSeparator()],
        [sg.Button('Export results CSV'), sg.Button('Compute distance'), sg.Button('Clear results')],
        [sg.Text('Results: (可多选两条记录以计算两点距离)') ],
        [sg.Listbox(values=[], size=(60,10), key='-RESULTS-', select_mode='extended', enable_events=True)],
    ]

    right_col = [
        [sg.Text('Video Frame (click to pick point):')],
        [sg.Graph(canvas_size=(900,600), graph_bottom_left=(0,0), graph_top_right=(900,600), key='-GRAPH-', enable_events=True)],
        [sg.Button('Prev'), sg.Button('Next'), sg.Button('Play/Pause'), sg.Text('', key='-PLAY_STATUS-')],
        [sg.Text('Click on frame to select a pixel -> will compute ground coords and predicted lon/lat (if origin available).')],
    ]

    layout = [[sg.Column(left_col), sg.VerticalSeparator(), sg.Column(right_col)]]
    return sg.Window('Homography Validator (pred lon/lat & export)', layout, finalize=True, resizable=True)

def run_app():
    window = build_window()
    cap = None
    total = 0
    idx = 0
    frame = None
    frame_scale = 1.0
    results = []  # list of dicts

    H = None
    origin_info = None   # (ox,oy,epsg)
    origin_lonlat = None # (lon, lat)
    playing = False

    def open_video(path):
        nonlocal cap, total, idx
        if cap is not None:
            cap.release()
            cap = None
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            sg.popup_error('无法打开视频: ' + path)
            return False
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        idx = 0
        window['-SLIDER-'].update(range=(0, max(0,total-1)))
        return True

    def read_frame(i):
        nonlocal cap
        if cap is None:
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ret, fr = cap.read()
        if not ret:
            return None
        return fr

    def update_display(fr):
        nonlocal frame_scale
        g = window['-GRAPH-']
        gw, gh = g.get_size()
        h,w = fr.shape[:2]
        sx = gw / float(w); sy = gh / float(h)
        frame_scale = min(sx, sy)
        disp = cv2.resize(fr, (int(w*frame_scale), int(h*frame_scale)))
        ret, buf = cv2.imencode('.png', disp)
        if not ret:
            return
        imgbytes = buf.tobytes()
        g.erase()
        g.draw_image(data=imgbytes, location=(0, gh))
        for i, r in enumerate(results):
            px = int(round(r['img_x'] * frame_scale)); py = int(round(r['img_y'] * frame_scale))
            g.draw_circle((px, gh-py), 5, fill_color='yellow')
            g.draw_text(str(i+1), (px+6, gh-py+6))

    while True:
        event, values = window.read(timeout=50)
        if event == sg.WIN_CLOSED:
            break

        if event == '-VIDEO-':
            path = values['-VIDEO-']
            if path and os.path.exists(path):
                ok = open_video(path)
                if ok:
                    fr = read_frame(0)
                    if fr is not None:
                        frame = fr
                        update_display(fr)
                        window['-FRAME-'].update('0')

        if event == 'Go':
            try: newi = int(values['-FRAME-'])
            except: newi = 0
            if cap is not None and 0 <= newi < total:
                idx = newi
                fr = read_frame(idx)
                if fr is not None:
                    frame = fr
                    update_display(fr)
                    window['-SLIDER-'].update(idx)

        if event == '-SLIDER-':
            try:
                val = int(values['-SLIDER-'])
            except:
                val = 0
            if cap is not None and val != idx:
                idx = val
                fr = read_frame(idx)
                if fr is not None:
                    frame = fr
                    update_display(fr)
                    window['-FRAME-'].update(idx)

        if event == 'Prev':
            if cap is not None:
                idx = max(0, idx-1)
                fr = read_frame(idx)
                if fr is not None:
                    frame = fr
                    update_display(fr)
                    window['-SLIDER-'].update(idx); window['-FRAME-'].update(idx)

        if event == 'Next':
            if cap is not None:
                idx = min(total-1, idx+1)
                fr = read_frame(idx)
                if fr is not None:
                    frame = fr
                    update_display(fr)
                    window['-SLIDER-'].update(idx); window['-FRAME-'].update(idx)

        if event == 'Play/Pause':
            playing = not playing
            window['-PLAY_STATUS-'].update('Playing' if playing else 'Paused')

        if playing and cap is not None:
            idx += 1
            if idx >= total:
                idx = 0
            fr = read_frame(idx)
            if fr is not None:
                frame = fr
                update_display(fr)
                window['-SLIDER-'].update(idx); window['-FRAME-'].update(idx)

        if event == 'Load H .npy':
            path = sg.popup_get_file('Select .npy file', file_types=(('NumPy','*.npy'),))
            if path:
                Hn = load_H_npy(path)
                if Hn is None:
                    sg.popup_error('无法读取 H（确保是 3x3 numpy 数组）')
                    window['-H_STATUS-'].update('H load failed')
                else:
                    H = Hn
                    window['-H_TEXT-'].update("\n".join([" ".join([f"{v:.12g}" for v in row]) for row in H.tolist()]))
                    window['-H_STATUS-'].update('H loaded')

        if event == 'Apply H':
            txt = values['-H_TEXT-']
            Hn = parse_H_text(txt)
            if Hn is None:
                sg.popup_error('无法解析 H，请输入 9 个数字（支持方括号、逗号、换行、科学计数法）或 Load .npy。')
                window['-H_STATUS-'].update('H parse failed')
            else:
                H = Hn
                window['-H_STATUS-'].update('H applied')
                print("Applied Homography H:\n", H)

        if event == '-GRAPH-':
            if frame is None:
                continue
            x,y = values['-GRAPH-']
            gw, gh = window['-GRAPH-'].get_size()
            img_x = int(round(x / frame_scale))
            img_y = int(round((gh - y) / frame_scale))
            if H is None:
                sg.popup('请先提供并 Apply 单应矩阵 H。')
                continue
            try:
                gx, gy = image_to_ground((img_x, img_y), H)
            except Exception as e:
                sg.popup_error('投影计算失败: ' + str(e))
                continue

            # Attempt to compute predicted lon/lat from gx,gy if we have origin_info or origin fields
            pred_lonlat = None
            try:
                if origin_info is None:
                    if values['-ORIGIN_LON-'].strip() != '' and values['-ORIGIN_LAT-'].strip() != '':
                        oy_lon = float(values['-ORIGIN_LON-']); oy_lat = float(values['-ORIGIN_LAT-'])
                        _, origin_info = lonlat_to_utm_local_single(oy_lon, oy_lat, (oy_lon, oy_lat))
                        origin_lonlat = (oy_lon, oy_lat)
                if origin_info is not None:
                    lon_pred, lat_pred = utm_local_to_lonlat(gx, gy, origin_info)
                    pred_lonlat = (lon_pred, lat_pred)
            except Exception:
                pred_lonlat = None

            pre_lon = f"{pred_lonlat[0]:.9f}" if pred_lonlat is not None else ''
            pre_lat = f"{pred_lonlat[1]:.9f}" if pred_lonlat is not None else ''

            form = [
                [sg.Text(f'Clicked pixel: ({img_x}, {img_y})')],
                [sg.Text(f'Predicted ground coords (by H): ({gx:.6f}, {gy:.6f})')],
            ]
            if pred_lonlat is not None:
                form.append([sg.Text(f'Predicted lon/lat (from H+origin): ({pre_lon}, {pre_lat})')])
            form += [
                [sg.Text('输入真实坐标：')],
                [sg.Radio('lon,lat (deg)', 'RAD', key='-R_LONLAT-', default=True),
                 sg.Radio('ground coords (meters, local)', 'RAD', key='-R_GND-')],
                [sg.Text('lon:'), sg.Input(pre_lon, size=(18,1), key='-IN_LON-'), sg.Text('lat:'), sg.Input(pre_lat, size=(18,1), key='-IN_LAT-')],
                [sg.Text('或直接输入地面坐标 x:'), sg.Input('', size=(12,1), key='-IN_GX-'), sg.Text('y:'), sg.Input('', size=(12,1), key='-IN_GY-')],
                [sg.Button('OK'), sg.Button('Cancel')]
            ]

            pop = sg.Window('Input true position (pred lon/lat shown if available)', form, modal=True)
            ev, vals2 = pop.read()
            pop.close()
            if ev != 'OK':
                continue

            use_lonlat = vals2['-R_LONLAT-']
            true_ground = None
            true_lonlat = None

            if use_lonlat:
                lon_txt = vals2['-IN_LON-'].strip()
                lat_txt = vals2['-IN_LAT-'].strip()
                if lon_txt == '' or lat_txt == '':
                    sg.popup_error('请选择 lon,lat 并输入值，或选择 ground coords。')
                    continue
                try:
                    lon_v = float(lon_txt); lat_v = float(lat_txt)
                    if values['-ORIGIN_LON-'].strip() == '' or values['-ORIGIN_LAT-'].strip() == '':
                        if Transformer is None:
                            sg.popup_error('左侧 Origin 为空且未安装 pyproj，无法自动设置 origin。请 pip install pyproj 或手动在左侧填写 Origin lon/lat，或直接输入地面坐标（meters）。')
                            continue
                        origin_lon = lon_v; origin_lat = lat_v
                        window['-ORIGIN_LON-'].update(f"{origin_lon:.12g}")
                        window['-ORIGIN_LAT-'].update(f"{origin_lat:.12g}")
                        _, origin_info = lonlat_to_utm_local_single(origin_lon, origin_lat, (origin_lon, origin_lat))
                        origin_lonlat = (origin_lon, origin_lat)
                        sg.popup('提示', '左侧 Origin 未填写，已将本次输入的 lon/lat 设为 origin（默认行为）。')
                    else:
                        origin_lon = float(values['-ORIGIN_LON-']); origin_lat = float(values['-ORIGIN_LAT-'])
                        if origin_info is None:
                            _, origin_info = lonlat_to_utm_local_single(origin_lon, origin_lat, (origin_lon, origin_lat))
                            origin_lonlat = (origin_lon, origin_lat)

                    (gx_true, gy_true), _ = lonlat_to_utm_local_single(lon_v, lat_v, (origin_lon, origin_lat))
                    true_ground = (gx_true, gy_true)
                    true_lonlat = (lon_v, lat_v)
                except Exception as e:
                    sg.popup_error('经纬度解析/转换失败: ' + str(e))
                    continue
            else:
                gx_txt = vals2['-IN_GX-'].strip(); gy_txt = vals2['-IN_GY-'].strip()
                if gx_txt == '' or gy_txt == '':
                    sg.popup_error('请输入地面坐标 x,y（meters）。')
                    continue
                try:
                    gx_true = float(gx_txt); gy_true = float(gy_txt)
                    true_ground = (gx_true, gy_true)
                except:
                    sg.popup_error('地面坐标解析失败。')
                    continue

            err = math.hypot(gx - true_ground[0], gy - true_ground[1])

            # compute pred lon/lat if possible (origin_info must be available)
            pred_lon_val = None; pred_lat_val = None
            if origin_info is not None:
                try:
                    lon_pred, lat_pred = utm_local_to_lonlat(gx, gy, origin_info)
                    pred_lon_val = float(lon_pred); pred_lat_val = float(lat_pred)
                except Exception:
                    pred_lon_val = None; pred_lat_val = None

            rec = {
                'frame': int(idx),
                'img_x': int(img_x),
                'img_y': int(img_y),
                'pred_gx': float(gx),
                'pred_gy': float(gy),
                'pred_lon': pred_lon_val,
                'pred_lat': pred_lat_val,
                'true_gx': float(true_ground[0]),
                'true_gy': float(true_ground[1]),
                'true_lon': (true_lonlat[0] if true_lonlat is not None else None),
                'true_lat': (true_lonlat[1] if true_lonlat is not None else None),
                'error_m': float(err)
            }
            results.append(rec)
            display_lines = []
            for i,r in enumerate(results):
                # keep display concise; pred lon/lat are in CSV
                line = f"{i+1:02d} frame={r['frame']} px=({r['img_x']},{r['img_y']}) pred=({r['pred_gx']:.3f},{r['pred_gy']:.3f}) true=({r['true_gx']:.3f},{r['true_gy']:.3f}) err={r['error_m']:.3f} m"
                display_lines.append(line)
            window['-RESULTS-'].update(display_lines)
            update_display(frame)

        if event == 'Compute distance':
            sel = values.get('-RESULTS-', [])
            if len(sel) != 2:
                sg.popup('请在结果列表中选中**两条**记录（多选），然后点击 "Compute distance"。')
            else:
                all_lines = window['-RESULTS-'].get_list_values()
                try:
                    idx1 = all_lines.index(sel[0])
                    idx2 = all_lines.index(sel[1])
                except ValueError:
                    sg.popup_error('选中项解析失败。')
                    continue
                r1 = results[idx1]; r2 = results[idx2]
                d_true = math.hypot(r1['true_gx'] - r2['true_gx'], r1['true_gy'] - r2['true_gy'])
                d_pred = math.hypot(r1['pred_gx'] - r2['pred_gx'], r1['pred_gy'] - r2['pred_gy'])
                sg.popup(f"两点距离（真实 ground）: {d_true:.3f} m\n两点距离（预测 pred）: {d_pred:.3f} m\n差异(pred-true): {(d_pred-d_true):.3f} m")

        if event == 'Export results CSV':
            if len(results) == 0:
                sg.popup('没有结果可导出')
            else:
                out = sg.popup_get_file('Save CSV', save_as=True, default_extension='csv', file_types=(('CSV','*.csv'),))
                if out:
                    try:
                        keys = ['frame','img_x','img_y','pred_gx','pred_gy','pred_lon','pred_lat','true_gx','true_gy','true_lon','true_lat','error_m']
                        with open(out, 'w', newline='', encoding='utf-8') as f:
                            w = csv.DictWriter(f, fieldnames=keys)
                            w.writeheader()
                            for r in results:
                                row = {k: r.get(k, '') for k in keys}
                                w.writerow(row)
                        sg.popup('Saved to', out)
                    except Exception as e:
                        sg.popup_error('保存失败: ' + str(e))

        if event == 'Clear results':
            results = []
            window['-RESULTS-'].update([])

    window.close()
    if cap is not None:
        cap.release()

if __name__ == '__main__':
    run_app()
