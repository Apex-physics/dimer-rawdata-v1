import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

# ================= 1. 终极防乱码：加载自带字体 =================
# 自动寻找代码同级目录下的 simhei.ttf 字体文件
font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    # 如果没找到字体文件（比如本地测试没放），兜底使用系统字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="量子多体动力学数据看板", layout="wide")

# ================= 2. 核心路径配置 =================
# 默认去寻找代码同级目录下的 Data 文件夹
DATA_DIR = os.path.join(os.path.dirname(__file__), "RawData")


# ================= 3. 纯内存动态扫盘 (无需 CSV) =================
@st.cache_data
def scan_data_folder():
    """纯内存扫描数据，不生成任何本地 CSV 文件"""
    records = []
    if not os.path.exists(DATA_DIR):
        return pd.DataFrame()

    for root, dirs, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        if not folder_name.startswith("L="):
            continue

        try:
            params = dict(item.split('=') for item in folder_name.split('_'))
            L = int(params.get('L', 0))
            init_state = params.get('Init', 'Unknown')
            freq = params.get('Freq', 'Unknown')
            U = float(params.get('U', 0.0))
            J = float(params.get('J', 0.0))
        except Exception:
            continue

        for file in files:
            if file.endswith('.npz') and file.startswith('SimData'):
                try:
                    eta_str = file.split('eta')[1].replace('.npz', '')
                    eta = float(eta_str)
                except Exception:
                    continue

                file_path = os.path.join(root, file)
                # 简单试读提取基础元数据
                try:
                    data = np.load(file_path, allow_pickle=True)
                    meta_dict = data['metadata'][0]
                    chi = int(meta_dict.get('chi_max', 512))
                    nmax = int(meta_dict.get('n_max', 3))
                    bc = 'OBC'
                    data.close()
                except Exception:
                    chi, nmax, bc = -1, -1, 'Error'

                records.append({
                    'L': L, 'Init': init_state, 'Freq': freq, 'U': U, 'J': J,
                    'eta': eta, 'chi': chi, 'nmax': nmax, 'bc': bc, 'file_path': file_path
                })

    if records:
        df = pd.DataFrame(records)
        return df.sort_values(by=['L', 'Init', 'Freq', 'U', 'eta']).reset_index(drop=True)
    return pd.DataFrame()


# ================= 4. 数据提取引擎 =================
@st.cache_data
def get_real_data(file_path, time_unit='t*J'):
    try:
        data = np.load(file_path, allow_pickle=True)
        times = data['times_tJ'] if time_unit == 't*J' else data['times_ms']
        occ_arr = data['occ_arr']
        P0_arr = data['P0_arr']
        P1_arr = data['P1_arr']
        P2_arr = data['P2_arr']
        err_prop = data['err_prop']
        data.close()
        return times, occ_arr, P0_arr, P1_arr, P2_arr, err_prop
    except Exception:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


def process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, L, obs_mode, site_or_range, metric):
    if len(times) == 0: return np.array([])
    center_idx = L // 2
    if obs_mode == "单格点":
        idx = site_or_range - 1
        if metric == "N": return occ_arr[:, idx]
        if metric == "P0": return P0_arr[:, idx]
        if metric == "P1": return P1_arr[:, idx]
        if metric == "P2": return P2_arr[:, idx]
    elif obs_mode == "局域范围":
        R = site_or_range
        start_idx = max(0, center_idx - R // 2)
        end_idx = min(L, center_idx + R // 2 + 1)

        region_N = occ_arr[:, start_idx:end_idx]
        odd_mask = np.array([(start_idx + i) % 2 == 0 for i in range(end_idx - start_idx)])
        even_mask = ~odd_mask

        if metric == "N全平均": return np.mean(region_N, axis=1)
        if metric == "N_odd平均": return np.mean(region_N[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(
            len(times))
        if metric == "N_even平均": return np.mean(region_N[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(
            len(times))
        if metric == "Imbalance":
            N_odd_avg = np.mean(region_N[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(len(times))
            N_even_avg = np.mean(region_N[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(len(times))
            return (N_odd_avg - N_even_avg) / (N_odd_avg + N_even_avg + 1e-9)

        # P0, P1, P2 简化展示逻辑
        region_P1 = P1_arr[:, start_idx:end_idx]
        if metric == "P1全平均": return np.mean(region_P1, axis=1)
    return np.zeros(len(times))


# ================= 5. 左侧栏：全局设置 =================
# 获取内存数据字典
df_registry = scan_data_folder()

if df_registry.empty:
    st.error("未找到数据。请检查 Data 文件夹是否上传，以及文件夹命名格式是否正确 (例如: L=17_Init=...)")
    st.stop()

st.sidebar.markdown("### 全局视图")
time_axis_unit = st.sidebar.radio("横坐标时间单位", ["t*J", "ms"])
obs_mode_global = st.sidebar.radio("观察区域模式", ["单格点", "局域范围"])
st.sidebar.markdown("---")

st.sidebar.markdown("### 物理参数筛选")
param_L = st.sidebar.selectbox("L", sorted(df_registry['L'].unique()))
df_f = df_registry[df_registry['L'] == param_L]

param_init = st.sidebar.selectbox("Init", sorted(df_f['Init'].unique()))
df_f = df_f[df_f['Init'] == param_init]

param_freq = st.sidebar.selectbox("Freq", sorted(df_f['Freq'].unique()))
df_f = df_f[df_f['Freq'] == param_freq]

param_U = st.sidebar.selectbox("U (Hz)", sorted(df_f['U'].unique()))
df_f = df_f[df_f['U'] == param_U]

param_J = st.sidebar.selectbox("J (Hz)", sorted(df_f['J'].unique()))
df_f = df_f[df_f['J'] == param_J]

param_eta = st.sidebar.selectbox("η", sorted(df_f['eta'].unique()))
df_f = df_f[df_f['eta'] == param_eta]

if df_f.empty:
    st.error("该条件下无数据。")
    st.stop()

current_data_row = df_f.iloc[0].to_dict()

# ================= 6. 主图绘制区 =================
st.title("量子多体动力学演化")

st.markdown("#### 输出对象配置")
col_cfg1, col_cfg2 = st.columns(2)

if obs_mode_global == "单格点":
    target_site = col_cfg1.number_input("格点索引 (真实序号 1~L)", min_value=1, max_value=param_L,
                                        value=(param_L // 2) + 1)
    target_metrics = col_cfg2.multiselect("输出物理量", ["N", "P0", "P1", "P2"], default=["N"])
    config_val = target_site
    label_prefix = f"格点 {target_site}"
else:
    range_opts = [i for i in range(1, param_L + 1) if i % 2 != 0]
    if param_L not in range_opts: range_opts.append(param_L)
    target_range = col_cfg1.selectbox("中心局域范围", range_opts, index=len(range_opts) - 1)
    target_metrics = col_cfg2.multiselect("输出物理量", ["N全平均", "N_odd平均", "N_even平均", "Imbalance", "P1全平均"],
                                          default=["N全平均"])
    config_val = target_range
    label_prefix = f"范围 {target_range}"

# 加载数据作图
times, occ_arr, P0_arr, P1_arr, P2_arr, err_prop = get_real_data(current_data_row['file_path'], time_axis_unit)

if len(times) > 0 and target_metrics:
    fig, ax = plt.subplots(figsize=(9, 4))
    for metric in target_metrics:
        y_data = process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, param_L, obs_mode_global, config_val,
                                     metric)
        # 这里的图例由于上面加载了 simhei，无论是本地还是云端都会完美显示中文！
        ax.plot(times, y_data, label=f"{label_prefix} - {metric}")

    ax.set_xlabel(f"时间 ({time_axis_unit})")
    ax.set_ylabel("观测值")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)
else:
    st.info("请选择物理量或检查数据文件。")