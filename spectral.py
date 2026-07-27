import os
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 1. 数据路径
# ============================================================

root_path = "/public/home/202411094934/python_project/Spatio-Temporal-Library/data/time-series-dataset/ETTm2"

train_data_path = os.path.join(root_path, "train_data.npy")
train_data_timestamps_path = os.path.join(root_path, "train_timestamps.npy")

val_data_path = os.path.join(root_path, "val_data.npy")
val_data_timestamps_path = os.path.join(root_path, "val_timestamps.npy")

test_data_path = os.path.join(root_path, "test_data.npy")
test_data_timestamps_path = os.path.join(root_path, "test_timestamps.npy")


# ============================================================
# 2. 加载数据
# ============================================================

train_data: np.ndarray = np.load(train_data_path)
train_data_timestamps: np.ndarray = np.load(
    train_data_timestamps_path
)

val_data: np.ndarray = np.load(val_data_path)
val_data_timestamps: np.ndarray = np.load(
    val_data_timestamps_path
)

test_data: np.ndarray = np.load(test_data_path)
test_data_timestamps: np.ndarray = np.load(
    test_data_timestamps_path
)

print("train_data shape:", train_data.shape)
print("val_data shape:", val_data.shape)
print("test_data shape:", test_data.shape)

print("train timestamps shape:", train_data_timestamps.shape)
print("val timestamps shape:", val_data_timestamps.shape)
print("test timestamps shape:", test_data_timestamps.shape)


# ============================================================
# 3. FFT 分析函数
# ============================================================

def analyze_frequency_phase(
    data: np.ndarray,
    feature_index: int = 0,
    channel_index: int = 0,
    start_index: int = 0,
    segment_length: int | None = None,
    sample_interval: float = 1.0,
    top_k: int = 10,
    use_window: bool = True,
    remove_mean: bool = True,
):
    """
    对指定时间窗口进行 FFT。

    按振幅从大到小选出 top_k 个非零频率分量，
    输出对应的频率、周期和相位。

    相位单位为弧度，通常位于 [-pi, pi]。
    """

    data = np.asarray(data)

    # ========================================================
    # 3.1 提取一维时间序列
    # ========================================================

    if data.ndim == 1:
        series = data

    elif data.ndim == 2:
        if not 0 <= feature_index < data.shape[1]:
            raise IndexError(
                f"feature_index={feature_index} 越界，"
                f"有效范围为 0 到 {data.shape[1] - 1}"
            )

        series = data[:, feature_index]

    elif data.ndim == 3:
        if not 0 <= feature_index < data.shape[1]:
            raise IndexError(
                f"feature_index={feature_index} 越界，"
                f"有效范围为 0 到 {data.shape[1] - 1}"
            )

        if not 0 <= channel_index < data.shape[2]:
            raise IndexError(
                f"channel_index={channel_index} 越界，"
                f"有效范围为 0 到 {data.shape[2] - 1}"
            )

        series = data[
            :,
            feature_index,
            channel_index,
        ]

    else:
        raise ValueError(
            "只支持形状为 (T,)、(T, N) 或 (T, N, C) 的数据，"
            f"当前形状为 {data.shape}"
        )

    series = np.asarray(
        series,
        dtype=np.float64,
    )

    # ========================================================
    # 3.2 截取指定窗口
    # ========================================================

    if start_index < 0:
        raise ValueError("start_index 不能小于 0。")

    if start_index >= len(series):
        raise ValueError(
            f"start_index={start_index} 超过序列长度 "
            f"{len(series)}。"
        )

    if segment_length is None:
        end_index = len(series)

    else:
        if segment_length < 2:
            raise ValueError(
                "segment_length 至少需要为 2。"
            )

        end_index = min(
            start_index + segment_length,
            len(series),
        )

    segment = series[start_index:end_index]

    if len(segment) < 2:
        raise ValueError(
            "实际截取的数据长度至少需要为 2。"
        )

    if not np.all(np.isfinite(segment)):
        raise ValueError(
            "截取的数据中包含 NaN 或 Inf。"
        )

    if sample_interval <= 0:
        raise ValueError(
            "sample_interval 必须大于 0。"
        )

    if top_k <= 0:
        raise ValueError(
            "top_k 必须大于 0。"
        )

    # ========================================================
    # 3.3 去均值
    # ========================================================

    if remove_mean:
        processed_segment = (
            segment - np.mean(segment)
        )
    else:
        processed_segment = segment.copy()

    n = len(processed_segment)

    # ========================================================
    # 3.4 加窗
    # ========================================================

    if use_window:
        window = np.hanning(n)
    else:
        window = np.ones(n)

    windowed_segment = (
        processed_segment * window
    )

    # ========================================================
    # 3.5 FFT
    # ========================================================

    fft_values = np.fft.rfft(
        windowed_segment
    )

    frequencies = np.fft.rfftfreq(
        n=n,
        d=sample_interval,
    )

    # 相位，单位为弧度
    phases = np.angle(fft_values)

    # ========================================================
    # 3.6 计算振幅，仅用于选择 Top-k
    # ========================================================

    window_sum = np.sum(window)

    if window_sum == 0:
        raise ValueError(
            "窗函数总和为 0。"
        )

    amplitudes = (
        2.0
        * np.abs(fft_values)
        / window_sum
    )

    # 直流分量不能乘 2
    amplitudes[0] = (
        np.abs(fft_values[0])
        / window_sum
    )

    # 偶数长度时，奈奎斯特频率不能乘 2
    if n % 2 == 0:
        amplitudes[-1] = (
            np.abs(fft_values[-1])
            / window_sum
        )

    # ========================================================
    # 3.7 排除频率为 0 的直流分量
    # ========================================================

    nonzero_mask = frequencies > 0

    nonzero_frequencies = frequencies[
        nonzero_mask
    ]

    nonzero_amplitudes = amplitudes[
        nonzero_mask
    ]

    nonzero_phases = phases[
        nonzero_mask
    ]

    # ========================================================
    # 3.8 按振幅选择 Top-k
    # ========================================================

    actual_top_k = min(
        top_k,
        len(nonzero_frequencies),
    )

    sorted_indices = np.argsort(
        nonzero_amplitudes
    )[::-1]

    top_indices = sorted_indices[
        :actual_top_k
    ]

    dominant_frequencies = (
        nonzero_frequencies[top_indices]
    )

    dominant_phases = (
        nonzero_phases[top_indices]
    )

    dominant_periods = (
        1.0 / dominant_frequencies
    )

    # ========================================================
    # 3.9 输出频率、周期和相位
    # ========================================================

    print(
        f"\nstart_index = {start_index}, "
        f"区间 = [{start_index}, {end_index})"
    )

    print("-" * 85)

    print(
        f"{'排名':<8}"
        f"{'频率':<22}"
        f"{'周期':<22}"
        f"{'相位(rad)':<22}"
    )

    print("-" * 85)

    for rank, (
        frequency,
        period,
        phase,
    ) in enumerate(
        zip(
            dominant_frequencies,
            dominant_periods,
            dominant_phases,
        ),
        start=1,
    ):
        print(
            f"{rank:<8}"
            f"{frequency:<22.10f}"
            f"{period:<22.6f}"
            f"{phase:<22.10f}"
        )

    return {
        "fft_values": fft_values,
        "frequencies": frequencies,
        "amplitudes": amplitudes,
        "phases": phases,
        "dominant_frequencies": dominant_frequencies,
        "dominant_periods": dominant_periods,
        "dominant_phases": dominant_phases,
        "start_index": start_index,
        "end_index": end_index,
    }


# # ============================================================
# # 4. 分析滑动窗口
# # ============================================================

# start_idxs = [0, 1, 2, 3, 4, 5]

# window_size = 128
# feature_index = 0

# train_results = {}

# for start_idx in start_idxs:
#     print(f"\n{'#' * 90}")
#     print(f"正在分析 start_idx = {start_idx}")
#     print(f"{'#' * 90}")

#     train_results[start_idx] = (
#         analyze_frequency_phase(
#             data=train_data,
#             feature_index=feature_index,
#             start_index=start_idx,
#             segment_length=window_size,
#             sample_interval=1.0,
#             top_k=10,
#             use_window=True,
#             remove_mean=True,
#         )
#     )

def extract_series(
    data: np.ndarray,
    feature_index: int = 0,
    channel_index: int = 0,
) -> np.ndarray:
    """
    从数据中提取一条一维时间序列。

    支持：
    (T,)
    (T, N)
    (T, N, C)
    """
    data = np.asarray(data)

    if data.ndim == 1:
        series = data

    elif data.ndim == 2:
        if not 0 <= feature_index < data.shape[1]:
            raise IndexError(
                f"feature_index={feature_index} 越界，"
                f"有效范围为 0 到 {data.shape[1] - 1}"
            )

        series = data[:, feature_index]

    elif data.ndim == 3:
        if not 0 <= feature_index < data.shape[1]:
            raise IndexError(
                f"feature_index={feature_index} 越界，"
                f"有效范围为 0 到 {data.shape[1] - 1}"
            )

        if not 0 <= channel_index < data.shape[2]:
            raise IndexError(
                f"channel_index={channel_index} 越界，"
                f"有效范围为 0 到 {data.shape[2] - 1}"
            )

        series = data[:, feature_index, channel_index]

    else:
        raise ValueError(
            "只支持形状为 (T,)、(T, N) 或 (T, N, C) 的数据，"
            f"当前形状为 {data.shape}"
        )

    series = np.asarray(series, dtype=np.float64)

    if not np.all(np.isfinite(series)):
        raise ValueError("数据中包含 NaN 或 Inf。")

    return series


def fourier_extrapolation(
    train_series: np.ndarray,
    future_length: int,
    sample_interval: float = 1.0,
    n_harmonics: int | None = None,
    detrend_type: str = "linear",
):
    """
    使用完整 train 序列的傅里叶频谱，对未来进行外推。

    参数
    ----------
    train_series:
        一维训练时间序列。

    future_length:
        需要预测的未来点数，例如 len(val) + len(test)。

    sample_interval:
        采样间隔。

    n_harmonics:
        使用多少个振幅最大的非零频率分量。

        None：
            使用所有频率分量。训练集可以近乎精确重建，
            未来相当于频谱的周期性延伸。

        例如 20、50、100：
            只保留主要频率，预测波形更加平滑。

    detrend_type:
        "linear"：
            去除线性趋势，FFT 后再把趋势外推回去。

        "constant"：
            只去均值。

        "none"：
            不去趋势。

    返回
    ----------
    result:
        包含完整预测、频率、振幅、相位、周期等信息。
    """
    train_series = np.asarray(
        train_series,
        dtype=np.float64,
    )

    if train_series.ndim != 1:
        raise ValueError("train_series 必须是一维数组。")

    if len(train_series) < 2:
        raise ValueError("train_series 长度至少为 2。")

    if future_length < 0:
        raise ValueError("future_length 不能小于 0。")

    if sample_interval <= 0:
        raise ValueError("sample_interval 必须大于 0。")

    n_train = len(train_series)

    train_index = np.arange(
        n_train,
        dtype=np.float64,
    )

    full_index = np.arange(
        n_train + future_length,
        dtype=np.float64,
    )

    # ========================================================
    # 1. 去趋势
    # ========================================================

    if detrend_type == "linear":
        trend_coefficients = np.polyfit(
            train_index,
            train_series,
            deg=1,
        )

        train_trend = np.polyval(
            trend_coefficients,
            train_index,
        )

        full_trend = np.polyval(
            trend_coefficients,
            full_index,
        )

    elif detrend_type == "constant":
        train_mean = float(np.mean(train_series))

        trend_coefficients = np.asarray(
            [train_mean],
            dtype=np.float64,
        )

        train_trend = np.full(
            n_train,
            train_mean,
            dtype=np.float64,
        )

        full_trend = np.full(
            n_train + future_length,
            train_mean,
            dtype=np.float64,
        )

    elif detrend_type == "none":
        trend_coefficients = np.asarray(
            [],
            dtype=np.float64,
        )

        train_trend = np.zeros(
            n_train,
            dtype=np.float64,
        )

        full_trend = np.zeros(
            n_train + future_length,
            dtype=np.float64,
        )

    else:
        raise ValueError(
            "detrend_type 只能是 "
            "'linear'、'constant' 或 'none'。"
        )

    detrended_train = (
        train_series - train_trend
    )

    # ========================================================
    # 2. 对完整 train 做 FFT
    # ========================================================

    fft_values = np.fft.rfft(
        detrended_train
    )

    frequencies = np.fft.rfftfreq(
        n=n_train,
        d=sample_interval,
    )

    phases = np.angle(
        fft_values
    )

    amplitudes = (
        2.0 * np.abs(fft_values) / n_train
    )

    # 直流分量不能乘 2
    amplitudes[0] = (
        np.abs(fft_values[0]) / n_train
    )

    # 偶数长度时，奈奎斯特频率不能乘 2
    if n_train % 2 == 0:
        amplitudes[-1] = (
            np.abs(fft_values[-1]) / n_train
        )

    periods = np.full(
        len(frequencies),
        np.inf,
        dtype=np.float64,
    )

    periods[1:] = (
        1.0 / frequencies[1:]
    )

    # ========================================================
    # 3. 选择需要用于外推的频率
    # ========================================================

    candidate_indices = np.arange(
        1,
        len(frequencies),
    )

    sorted_indices = candidate_indices[
        np.argsort(
            amplitudes[candidate_indices]
        )[::-1]
    ]

    if n_harmonics is None:
        selected_indices = sorted_indices
    else:
        if n_harmonics <= 0:
            raise ValueError(
                "n_harmonics 必须大于 0 或为 None。"
            )

        selected_indices = sorted_indices[
            :min(n_harmonics, len(sorted_indices))
        ]

    # ========================================================
    # 4. 将频率分量延伸到 train + val + test
    # ========================================================

    full_time = (
        full_index * sample_interval
    )

    reconstructed = (
        full_trend.copy()
    )

    # 加回直流分量
    reconstructed += (
        fft_values[0].real / n_train
    )

    for frequency_index in selected_indices:
        frequency = frequencies[
            frequency_index
        ]

        amplitude = amplitudes[
            frequency_index
        ]

        phase = phases[
            frequency_index
        ]

        reconstructed += (
            amplitude
            * np.cos(
                2.0
                * np.pi
                * frequency
                * full_time
                + phase
            )
        )

    return {
        "prediction": reconstructed,
        "train_reconstruction": reconstructed[
            :n_train
        ],
        "future_prediction": reconstructed[
            n_train:
        ],
        "fft_values": fft_values,
        "frequencies": frequencies,
        "periods": periods,
        "amplitudes": amplitudes,
        "phases": phases,
        "selected_indices": selected_indices,
        "detrended_train": detrended_train,
        "train_trend": train_trend,
        "full_trend": full_trend,
        "trend_coefficients": trend_coefficients,
        "n_harmonics": n_harmonics,
        "detrend_type": detrend_type,
    }


def calculate_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict:
    """
    计算 MAE、MSE、RMSE。
    """
    actual = np.asarray(
        actual,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    if actual.shape != predicted.shape:
        raise ValueError(
            f"actual 形状 {actual.shape} 与 "
            f"predicted 形状 {predicted.shape} 不一致。"
        )

    error = predicted - actual

    mae = np.mean(
        np.abs(error)
    )

    mse = np.mean(
        error ** 2
    )

    rmse = np.sqrt(mse)

    max_absolute_error = np.max(
        np.abs(error)
    )

    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MaxAE": max_absolute_error,
    }


def print_metrics(
    name: str,
    metrics: dict,
):
    print(f"\n{name}")
    print("-" * 60)
    print(f"MAE   : {metrics['MAE']:.10f}")
    print(f"MSE   : {metrics['MSE']:.10f}")
    print(f"RMSE  : {metrics['RMSE']:.10f}")
    print(f"MaxAE : {metrics['MaxAE']:.10f}")


# ============================================================
# 3. 提取 train、val、test 的同一个变量
# ============================================================

feature_index = 2

train_series = extract_series(
    train_data,
    feature_index=feature_index,
)

val_series = extract_series(
    val_data,
    feature_index=feature_index,
)

test_series = extract_series(
    test_data,
    feature_index=feature_index,
)

n_train = len(train_series)
n_val = len(val_series)
n_test = len(test_series)

future_length = n_val + n_test

print("\n数据长度")
print("-" * 60)
print("train length:", n_train)
print("val length  :", n_val)
print("test length :", n_test)
print("future length:", future_length)


# ============================================================
# 4. 用整个 train 频谱外推 val 和 test
# ============================================================

fourier_result = fourier_extrapolation(
    train_series=train_series,
    future_length=future_length,
    sample_interval=1.0,

    # None 表示使用所有频率分量
    n_harmonics=None,

    # 可以改成 "constant" 进行对比
    detrend_type="linear",
)

full_prediction = fourier_result[
    "prediction"
]

train_reconstruction = fourier_result[
    "train_reconstruction"
]

future_prediction = fourier_result[
    "future_prediction"
]

val_prediction = future_prediction[
    :n_val
]

test_prediction = future_prediction[
    n_val:n_val + n_test
]


# ============================================================
# 5. 计算误差
# ============================================================

train_metrics = calculate_metrics(
    actual=train_series,
    predicted=train_reconstruction,
)

val_metrics = calculate_metrics(
    actual=val_series,
    predicted=val_prediction,
)

test_metrics = calculate_metrics(
    actual=test_series,
    predicted=test_prediction,
)

print_metrics(
    "Train reconstruction metrics",
    train_metrics,
)

print_metrics(
    "Validation forecast metrics",
    val_metrics,
)

print_metrics(
    "Test forecast metrics",
    test_metrics,
)


# ============================================================
# 6. 打印 train 的主要频率信息
# ============================================================

frequencies = fourier_result[
    "frequencies"
]

periods = fourier_result[
    "periods"
]

amplitudes = fourier_result[
    "amplitudes"
]

phases = fourier_result[
    "phases"
]

top_k = 10

nonzero_indices = np.arange(
    1,
    len(frequencies),
)

top_indices = nonzero_indices[
    np.argsort(
        amplitudes[nonzero_indices]
    )[::-1]
][:top_k]

print("\n完整 train 的主要频率成分")
print("-" * 105)

print(
    f"{'排名':<8}"
    f"{'FFT bin':<12}"
    f"{'频率':<22}"
    f"{'周期':<22}"
    f"{'振幅':<20}"
    f"{'相位(rad)':<20}"
)

print("-" * 105)

for rank, frequency_index in enumerate(
    top_indices,
    start=1,
):
    print(
        f"{rank:<8}"
        f"{frequency_index:<12}"
        f"{frequencies[frequency_index]:<22.10f}"
        f"{periods[frequency_index]:<22.6f}"
        f"{amplitudes[frequency_index]:<20.10f}"
        f"{phases[frequency_index]:<20.10f}"
    )


# ============================================================
# 7. 绘制 train、val、test 及预测波形
# ============================================================

train_x = np.arange(
    0,
    n_train,
)

val_x = np.arange(
    n_train,
    n_train + n_val,
)

test_x = np.arange(
    n_train + n_val,
    n_train + n_val + n_test,
)

# 第一张图：完整数据范围
plt.figure(
    figsize=(18, 7)
)

plt.plot(
    train_x,
    train_series,
    label="Train actual",
    linewidth=1.0,
)

plt.plot(
    val_x,
    val_series,
    label="Validation actual",
    linewidth=1.0,
)

plt.plot(
    test_x,
    test_series,
    label="Test actual",
    linewidth=1.0,
)

plt.plot(
    val_x,
    val_prediction,
    label="Validation Fourier prediction",
    linestyle="--",
    linewidth=1.2,
)

plt.plot(
    test_x,
    test_prediction,
    label="Test Fourier prediction",
    linestyle="--",
    linewidth=1.2,
)

plt.axvline(
    n_train,
    linestyle="--",
    linewidth=1.0,
)

plt.axvline(
    n_train + n_val,
    linestyle="--",
    linewidth=1.0,
)

plt.xlabel("Time index")
plt.ylabel("Value")

plt.title(
    "Fourier Extrapolation: "
    "Train Spectrum to Validation and Test"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 8. 只显示 train 尾部、val 和 test
# ============================================================

history_length = min(
    512,
    n_train,
)

history_start = (
    n_train - history_length
)

plt.figure(
    figsize=(18, 7)
)

plt.plot(
    train_x[history_start:],
    train_series[history_start:],
    label="Train actual tail",
    linewidth=1.0,
)

plt.plot(
    val_x,
    val_series,
    label="Validation actual",
    linewidth=1.2,
)

plt.plot(
    val_x,
    val_prediction,
    label="Validation Fourier prediction",
    linestyle="--",
    linewidth=1.2,
)

plt.plot(
    test_x,
    test_series,
    label="Test actual",
    linewidth=1.2,
)

plt.plot(
    test_x,
    test_prediction,
    label="Test Fourier prediction",
    linestyle="--",
    linewidth=1.2,
)

plt.axvline(
    n_train,
    linestyle="--",
    linewidth=1.0,
    label="Validation start",
)

plt.axvline(
    n_train + n_val,
    linestyle="--",
    linewidth=1.0,
    label="Test start",
)

plt.xlabel("Time index")
plt.ylabel("Value")

plt.title(
    "Actual vs Fourier-Predicted "
    "Validation and Test Waveforms"
)

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 9. 分别绘制 val 和 test 误差
# ============================================================

plt.figure(
    figsize=(18, 5)
)

plt.plot(
    val_x,
    val_prediction - val_series,
    label="Validation error",
    linewidth=1.0,
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

plt.xlabel("Time index")
plt.ylabel("Prediction - Actual")
plt.title("Validation Fourier Extrapolation Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


plt.figure(
    figsize=(18, 5)
)

plt.plot(
    test_x,
    test_prediction - test_series,
    label="Test error",
    linewidth=1.0,
)

plt.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)

plt.xlabel("Time index")
plt.ylabel("Prediction - Actual")
plt.title("Test Fourier Extrapolation Error")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()