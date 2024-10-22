import numpy as np
import statsmodels.api as sm
from filterpy.kalman import KalmanFilter

def calculate_trend(x, y):
    x = sm.add_constant(x)
    model = sm.OLS(y, x)
    results = model.fit()
    return results.params[1]

def get_color(value):
    if value < 31:
        return np.array([8, 171, 29]) / 255
    if value < 71:
        return np.array([255, 255, 0]) / 255
    return np.array([179, 19, 19]) / 255

def unit_vector(v):
    return v / np.linalg.norm(v)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def calc_dispersion_old(gvs):
    gv_mean = unit_vector(np.sum(gvs, axis=0))
    angle = 0
    for gv in gvs:
        angle += angle_between(gv_mean, gv)
    return angle / len(gvs)

def calc_dispersion(gvs):
    gv_mean = unit_vector(np.mean(gvs, axis=0))
    gvs_u = gvs / np.linalg.norm(gvs, axis=1)[:, np.newaxis]
    return np.mean(np.rad2deg(np.arccos(np.clip(np.sum(gvs_u * gv_mean, axis=1), -1.0, 1.0))))

def calc_dispersion_index(gvs):
    gvs_u = gvs / np.linalg.norm(gvs, axis=1)[:, np.newaxis]
    gv_mean = np.mean(gvs_u, axis=0)
    return np.linalg.norm(gv_mean)

def event_detection(gvs, timestamps, th_dispersion, th_duration_max, th_duration_min):
    fixations = list()
    current_window = list()
    start_idx = 0
    t0 = timestamps[0]
    for i, (t, gv) in enumerate(zip(timestamps, gvs)):
        current_window.append(gv)
        dispersion = calc_dispersion(current_window) if len(current_window) > 1 else 0
        duration = t - t0
        if dispersion > th_dispersion or duration > th_duration_max:
            if len(current_window) > 1 and duration > th_duration_min:
                fixations.append({
                    "StartIdx": start_idx,
                    "EndIdx": i-1,
                    "Duration": duration,
                    "StartTimestamp": t0,
                    "EndTimestamp": t
                })
            current_window = [gv]
            start_idx = i
            t0 = t
    return fixations

def use_kalman_filter(gvs):
    kf = KalmanFilter(dim_x=3, dim_z=3)
    kf.x = gvs[:1].reshape(3, 1)
    kf.H = np.eye(3)
    kf.P = np.eye(3) * 0.1
    kf.Q = np.eye(3) * 0.1
    kf.R = np.eye(3) * 0.1
    gvs_filtered = np.empty_like(gvs)
    for i in range(len(gvs)):
        kf.predict()
        kf.update(gvs[i])
        gvs_filtered[i] = kf.x.copy().squeeze()
    return gvs_filtered


if __name__=="__main__":
    # from data import *
    # rec = Recording("Data/TestRecordings/ALena")
    # gvs = rec["nod"]["Gaze120"].loc[rec["nod"]["Gaze120"].Message=="gaze sample", ["Local Gaze Direction %s" % x for x in ["X", "Y", "Z"]]].to_numpy()
    # t = rec["nod"]["Gaze120"].loc[rec["nod"]["Gaze120"].Message=="gaze sample", "Device Timestamp"].to_numpy()
    # fixations = event_detection(gvs, t, 2, 300, 50)
    # for i, fix in enumerate(fixations):
    #     print(i, fix)
    x = np.arange(10)
    y = np.random.rand(10)
    print(calculate_trend(y, x))
    