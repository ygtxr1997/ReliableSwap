import cv2
import numpy as np
from scipy.signal import savgol_filter

def kalman_filter(inputs: np.array,
                  process_noise: float = 0.03,
                  measure_noise: float = 0.01,
                  ):
    """ OpenCV - Kalman Filter
    https://blog.csdn.net/angelfish91/article/details/61768575
    https://blog.csdn.net/qq_23981335/article/details/82968422
    """
    assert inputs.ndim == 2, "inputs should be 2-dim np.array"

    '''
    它有3个输入参数，
    dynam_params：状态空间的维数，这里为2；
    measure_param：测量值的维数，这里也为2; 
    control_params：控制向量的维数，默认为0。由于这里该模型中并没有控制变量，因此也为0。
    '''
    kalman = cv2.KalmanFilter(2,2)

    kalman.measurementMatrix = np.array([[1,0],[0,1]],np.float32)
    kalman.transitionMatrix = np.array([[1,0],[0,1]], np.float32)
    kalman.processNoiseCov = np.array([[1,0],[0,1]], np.float32) * process_noise
    kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * measure_noise
    '''
    kalman.measurementNoiseCov为测量系统的协方差矩阵，方差越小，预测结果越接近测量值，
    kalman.processNoiseCov为模型系统的噪声，噪声越大，预测结果越不稳定，越容易接近模型系统预测值，且单步变化越大，
    相反，若噪声小，则预测结果与上个计算结果相差不大。
    '''

    kalman.statePre = np.array([[inputs[0][0]],
                                [inputs[0][1]]])

    '''
    Kalman Filtering
    '''
    outputs = np.zeros_like(inputs)
    for i in range(len(inputs)):
        mes = np.reshape(inputs[i,:],(2,1))

        x = kalman.correct(mes)

        y = kalman.predict()
        outputs[i] = np.squeeze(y)
        # print (kalman.statePost[0],kalman.statePost[1])
        # print (kalman.statePre[0],kalman.statePre[1])
        # print ('measurement:\t',mes[0],mes[1])
        # print ('correct:\t',x[0],x[1])
        # print ('predict:\t',y[0],y[1])
        # print ('='*30)

    return outputs


def kalman_filter_landmark(landmarks: np.array,
                           process_noise: float = 0.03,
                           measure_noise: float = 0.01,
                           ):
    """ Kalman Filter for Landmarks
    :param process_noise: large means unstable and close to model predictions
    :param measure_noise: small means close to measurement
    """
    print('[Using Kalman Filter for Landmark Smoothing, process_noise=%f, measure_noise=%f]' %
          (process_noise, measure_noise))

    ''' 
    landmarks: (#frames, key, xy)
    '''
    assert landmarks.ndim == 3, 'landmarks should be 3-dim np.array'
    assert landmarks.dtype == 'float32', 'landmarks dtype should be float32'

    for s1 in range(landmarks.shape[1]):
        landmarks[:, s1] = kalman_filter(landmarks[:, s1],
                                         process_noise,
                                         measure_noise)
    return landmarks


def savgol_filter_landmark(landmarks: np.array,
                           window_length: int = 25,
                           poly_order: int = 2,
                           ):
    """ Savgol Filter for Landmarks
    https://blog.csdn.net/kaever/article/details/105520941
    """
    print('[Using Savgol Filter for Landmark Smoothing, window_length=%d, poly_order=%d]' %
          (window_length, poly_order))

    ''' 
    landmarks: (#frames, key, xy)
    '''
    assert landmarks.ndim == 3, 'landmarks should be 3-dim np.array'
    assert landmarks.dtype == 'float32', 'landmarks dtype should be float32'
    assert window_length % 2 == 1, 'window_length should be odd'

    for s1 in range(landmarks.shape[1]):
        for s2 in range(landmarks.shape[2]):
            landmarks[:, s1, s2] = savgol_filter(landmarks[:, s1, s2],
                                                 window_length,
                                                 poly_order)
    return landmarks

if __name__ == '__main__':

    pos = np.array([
            [10,    50],
            [12,    49],
            [11,    52],
            [13,    52.2],
            [12.9,  50]], np.float32)

    print(pos)
    pos_filtered = kalman_filter(pos)
    print(pos)
    print(pos_filtered)