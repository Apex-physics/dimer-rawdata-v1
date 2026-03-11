import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import datetime

# ================= 解决 Matplotlib 中文乱码问题 =================
font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="量子多体动力学数据看板", layout="wide")

# ================= 核心配置区 =================
DATA_DIR = os.path.join(os.path.dirname(__file__), "RawData")
REGISTRY_FILE = "file_registry.csv"


# ================= 1. 数据扫描与建库逻辑 =================
def scan_and_build_registry():
    """扫描文件夹并重新生成 CSV 映射表"""
    records = []
    if not os.path.exists(DATA_DIR):
        return False, f"找不到数据目录: {DATA_DIR}"

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
                    eta_str = file.split('eta=')[1].split('_')[0]
                    eta = float(eta_str)
                except Exception:
                    eta = 0.0

                file_path = os.path.join(root, file)
                try:
                    data = np.load(file_path, allow_pickle=True)
                    meta_dict = data['metadata'][0]

                    if 'chi=' in file:
                        chi_str = file.split('chi=')[-1].replace('.npz', '')
                        chi = int(chi_str)
                    else:
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
        df = df.sort_values(by=['L', 'Init', 'Freq', 'U', 'eta', 'chi']).reset_index(drop=True)
        df.to_csv(REGISTRY_FILE, index=False)
        return True, len(df)
    else:
        return False, "未找到任何有效数据"


@st.cache_data
def load_registry(last_refresh_time):
    if os.path.exists(REGISTRY_FILE):
        return pd.read_csv(REGISTRY_FILE)
    return pd.DataFrame()


# ================= 2. 真实数据读取与处理引擎 =================
@st.cache_data
def get_real_data(file_path, time_unit='t*J'):
    """读取真实的 npz 数据包"""
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
    except Exception as e:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])


def process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, L, obs_mode, site_or_range, metric):
    """根据选项切片并计算物理量"""
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
        region_P0 = P0_arr[:, start_idx:end_idx]
        region_P1 = P1_arr[:, start_idx:end_idx]
        region_P2 = P2_arr[:, start_idx:end_idx]

        odd_mask = np.array([(start_idx + i) % 2 != 0 for i in range(end_idx - start_idx)])
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

        if metric == "P0全平均": return np.mean(region_P0, axis=1)
        if metric == "P0_odd平均": return np.mean(region_P0[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(
            len(times))
        if metric == "P0_even平均": return np.mean(region_P0[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(
            len(times))

        if metric == "P1全平均": return np.mean(region_P1, axis=1)
        if metric == "P1_odd平均": return np.mean(region_P1[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(
            len(times))
        if metric == "P1_even平均": return np.mean(region_P1[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(
            len(times))

        if metric == "P2全平均": return np.mean(region_P2, axis=1)
        if metric == "P2_odd平均": return np.mean(region_P2[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(
            len(times))
        if metric == "P2_even平均": return np.mean(region_P2[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(
            len(times))

    return np.zeros(len(times))


def apply_truncation(times, y_data, error, cutoff_mode, custom_err_limit=None):
    if cutoff_mode == "自定义误差截断" and custom_err_limit is not None:
        exceed_indices = np.where(error > custom_err_limit)[0]
        if len(exceed_indices) > 0:
            idx = exceed_indices[0]
            return times[:idx], y_data[:idx]
    return times, y_data


# ================= 对比池初始化 =================
if 'compare_lines' not in st.session_state:
    st.session_state.compare_lines = []
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") if os.path.exists(
        REGISTRY_FILE) else "从未刷新"

# ================= 3. 左侧栏 =================
st.sidebar.markdown("### 数据字典维护")
if st.sidebar.button("扫描硬盘刷新数据"):
    with st.spinner("正在解析 npz 文件..."):
        success, msg = scan_and_build_registry()
        if success:
            st.session_state['last_refresh'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.sidebar.success(f"刷新成功！载入 {msg} 个文件。")
        else:
            st.sidebar.error(msg)
st.sidebar.caption(f"最后刷新: {st.session_state['last_refresh']}")
st.sidebar.markdown("---")

df_registry = load_registry(st.session_state['last_refresh'])
if df_registry.empty:
    st.error("数据字典为空，请在左侧点击【扫描硬盘刷新数据】。")
    st.stop()

st.sidebar.markdown("### 全局视图")
time_axis_unit = st.sidebar.radio("横坐标时间单位", ["t*J", "ms"])
obs_mode_global = st.sidebar.radio("观察区域模式", ["单格点", "局域范围"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 物理参数筛选 (单组)")
param_L = st.sidebar.selectbox("系统尺寸 (L)", sorted(df_registry['L'].unique()))
df_f = df_registry[df_registry['L'] == param_L]

param_init = st.sidebar.selectbox("初始构型 (Init)", sorted(df_f['Init'].unique()))
df_f = df_f[df_f['Init'] == param_init]

param_freq = st.sidebar.selectbox("驱动频率 (Freq)", sorted(df_f['Freq'].unique()))
df_f = df_f[df_f['Freq'] == param_freq]

param_U = st.sidebar.selectbox("U (Hz)", sorted(df_f['U'].unique()))
df_f = df_f[df_f['U'] == param_U]

param_J = st.sidebar.selectbox("J (Hz)", sorted(df_f['J'].unique()))
df_f = df_f[df_f['J'] == param_J]

param_eta = st.sidebar.selectbox("驱动强度 (η)", sorted(df_f['eta'].unique()))
df_f = df_f[df_f['eta'] == param_eta]

# 【修改点】侧边栏选择 chi 时，主动过滤掉 chi=700
available_chis = sorted(df_f[df_f['chi'] != 700]['chi'].unique(), reverse=True)
if not available_chis:
    st.error("该实验条件下除基准(chi=700)外无其他数据。")
    st.stop()
param_chi = st.sidebar.selectbox("截断维数 (χ)", available_chis)
df_f_final = df_f[df_f['chi'] == param_chi]

if df_f_final.empty:
    st.error("该实验条件下暂时没有数据。")
    st.stop()

current_data_row = df_f_final.iloc[0].to_dict()

# ================= 4. 顶部：信息板 =================
st.title("量子多体动力学看板")
col_algo, col_chi, col_nmax, col_bc = st.columns(4)
col_algo.metric("算法", "TEBD (TeNPy)")
col_chi.metric("当前显示 χ", current_data_row['chi'])
col_nmax.metric("局域玻色子 (n_max)", current_data_row['nmax'])
col_bc.metric("边界条件", current_data_row['bc'])
st.markdown("---")

# ================= 5. 主展区 =================
col_main, col_side = st.columns([3, 1])

with col_main:
    tab_single, tab_compare = st.tabs(["单组数据探索", "跨条件自由构建对比"])

    with tab_single:
        st.markdown("#### 输出对象配置")
        col_cfg1, col_cfg2 = st.columns(2)

        if obs_mode_global == "单格点":
            target_site = col_cfg1.number_input("格点索引", min_value=1, max_value=param_L, value=(param_L // 2) + 1)
            target_metrics = col_cfg2.multiselect("输出物理量", ["N", "P0", "P1", "P2"], default=["N"])
            config_val = target_site
            label_prefix = f"格点 {target_site}"
        else:
            range_opts = [i for i in range(1, param_L + 1) if i % 2 != 0]
            if param_L not in range_opts: range_opts.append(param_L)
            target_range = col_cfg1.selectbox("中心局域范围", range_opts, index=len(range_opts) - 1)
            metric_opts = ["N全平均", "N_odd平均", "N_even平均", "Imbalance", "P0全平均", "P0_odd平均", "P0_even平均",
                           "P1全平均", "P1_odd平均", "P1_even平均", "P2全平均", "P2_odd平均", "P2_even平均"]
            target_metrics = col_cfg2.multiselect("输出物理量", metric_opts, default=["N全平均"])
            config_val = target_range
            label_prefix = f"范围 {target_range}"

        st.markdown("---")
        c_trunc1, c_trunc2 = st.columns([1, 2])
        single_cutoff_mode = c_trunc1.radio("误差截断模式", ["无截断 (全长)", "自定义误差截断"], horizontal=True)
        single_err_limit = c_trunc2.number_input("传递误差阈值 (%)", value=2.0,
                                                 step=0.5) if single_cutoff_mode == "自定义误差截断" else None

        times, occ_arr, P0_arr, P1_arr, P2_arr, err_prop = get_real_data(current_data_row['file_path'], time_axis_unit)

        if len(times) > 0 and target_metrics:
            fig, ax = plt.subplots(figsize=(9, 4))
            for metric in target_metrics:
                y_data = process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, param_L, obs_mode_global,
                                             config_val, metric)
                t_plot, y_plot = apply_truncation(times, y_data, err_prop, single_cutoff_mode, single_err_limit)
                ax.plot(t_plot, y_plot, label=f"{label_prefix} - {metric}")
            ax.set_xlabel(f"Time ({time_axis_unit})")
            ax.set_ylabel("Value");
            ax.grid(True, alpha=0.3);
            ax.legend()
            st.pyplot(fig)
        elif len(times) == 0:
            st.error("数据读取失败。")
        else:
            st.info("请选择物理量。")

    with tab_compare:
        st.markdown("#### 自由对比曲线构建器")
        all_exp_strs = []
        df_registry_sorted = df_registry.sort_values(by=['L', 'Init', 'Freq', 'U', 'J', 'eta', 'chi'])
        for _, row in df_registry_sorted.iterrows():
            all_exp_strs.append(
                f"L={row['L']} | {row['Init']} | F={row['Freq']} | U={row['U']} | J={row['J']} | η={row['eta']} | χ={row['chi']}")

        b_exp_str = st.selectbox("[1] 选择基础物理条件 (包含 χ)", all_exp_strs)
        b_idx = all_exp_strs.index(b_exp_str);
        b_row_data = df_registry_sorted.iloc[b_idx].to_dict()
        b_L = b_row_data['L']

        c4, c5, c6 = st.columns(3)
        b_obs_mode = c4.radio("[2] 观察区域", ["单格点", "局域范围"], horizontal=True, key="b_obs_mode")
        if b_obs_mode == "单格点":
            b_site_or_range = c5.number_input("[3] 格点数", min_value=1, max_value=b_L, value=b_L // 2 + 1,
                                              key="b_site")
            b_metric_opts = ["和现有输出一致", "N", "P0", "P1", "P2"]
            prefix_desc = f"格点 {b_site_or_range}"
        else:
            b_range_opts = [i for i in range(1, b_L + 1) if i % 2 != 0]
            if b_L not in b_range_opts: b_range_opts.append(b_L)
            b_site_or_range = c5.selectbox("[3] 局域范围", b_range_opts, index=len(b_range_opts) - 1, key="b_range")
            b_metric_opts = ["和现有输出一致", "N全平均", "N_odd平均", "N_even平均", "Imbalance", "P0全平均",
                             "P0_odd平均", "P0_even平均", "P1全平均", "P1_odd平均", "P1_even平均", "P2全平均",
                             "P2_odd平均", "P2_even平均"]
            prefix_desc = f"范围 {b_site_or_range}"

        b_metric = c6.selectbox("[4] 物理量", b_metric_opts, key="b_metric")

        if st.button("将该曲线加入对比池"):
            m_list = [b_metric] if b_metric != "和现有输出一致" else target_metrics
            for m in m_list:
                st.session_state.compare_lines.append({
                    'desc': f"{b_exp_str.replace(' | ', ', ')} | {prefix_desc} {m}",
                    'file_path': b_row_data['file_path'], 'L': b_L, 'obs_mode': b_obs_mode,
                    'site_or_range': b_site_or_range, 'metric': m
                })
                st.success(f"已加入: {m}")

# ================= 6. 右侧：截断收敛性展示 (重新详细实现的逻辑) =================
with col_side:
    st.markdown("### 💠 χ 收敛性分析")

    # 【逻辑1】数据选择逻辑：寻找同 L, 同 Init 且 chi=700 的对比数据
    # 按照您的要求，寻找与当前所选物理参数完全匹配的 chi=700 样本
    ref_query = df_registry[
        (df_registry['L'] == param_L) &
        (df_registry['Init'] == param_init) &
        (df_registry['Freq'] == param_freq) &
        (df_registry['U'] == param_U) &
        (df_registry['J'] == param_J) &
        (df_registry['eta'] == param_eta) &
        (df_registry['chi'] == 700)
        ]

    if ref_query.empty:
        st.write("📊 **对比状态**")
        st.info("当前物理参数组合下暂无对应的 chi=700 基准数据。")
    else:
        # 【逻辑2】测量量选取：以单组数据最后选择的为准，若无则默认 N_even平均 全格点
        if target_metrics:
            comp_metric = target_metrics[-1]
            c_mode, c_val = obs_mode_global, config_val
        else:
            comp_metric = "N_even平均"
            c_mode, c_val = "局域范围", param_L  # 默认全格点范围

        ref_row = ref_query.iloc[0].to_dict()
        t_ref, o_ref, p0_ref, p1_ref, p2_ref, _ = get_real_data(ref_row['file_path'], time_axis_unit)
        y_ref = process_target_data(t_ref, o_ref, p0_ref, p1_ref, p2_ref, param_L, c_mode, c_val, comp_metric)

        # 当前显示数据
        y_cur = process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, param_L, c_mode, c_val, comp_metric)

        # 绘图
        fig_cv, ax_cv = plt.subplots(figsize=(4, 3.5))
        ax_cv.plot(t_ref, y_ref, 'k--', label="χ=700 (基准)", alpha=0.7)
        ax_cv.plot(times, y_cur, 'r-', label=f"χ={param_chi}", linewidth=1)
        ax_cv.set_title(f"对比量: {comp_metric}", fontsize=9)
        ax_cv.tick_params(labelsize=8);
        ax_cv.legend(fontsize=7);
        ax_cv.grid(True, alpha=0.2)
        st.pyplot(fig_cv)

        # 【逻辑3】误差时间判定逻辑：输出在多少时误差达到 1%
        min_len = min(len(y_ref), len(y_cur))
        if min_len > 0:
            diff = np.abs(y_ref[:min_len] - y_cur[:min_len])
            # 相对误差：以基准值的最大绝对值为归一化因子
            norm = np.max(np.abs(y_ref)) if np.max(np.abs(y_ref)) > 1e-9 else 1.0
            rel_err = diff / norm

            first_fail_idx = np.where(rel_err >= 0.01)[0]
            if len(first_fail_idx) > 0:
                fail_time = times[first_fail_idx[0]]
                st.write(f"在 {fail_time:.2f} {time_axis_unit} 时误差达到 1%")
            else:
                st.write("全时间段内误差未达到 1%")