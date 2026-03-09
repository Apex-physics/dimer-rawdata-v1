import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import datetime

# ================= 解决 Matplotlib 中文乱码问题 =================
# 自动寻找代码同级目录下的 simhei.ttf 字体文件，解决云端 Linux 没有中文字体的问题
font_path = os.path.join(os.path.dirname(__file__), "simhei.ttf")
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
else:
    # 如果没找到字体文件（比如在本地运行），兜底使用系统字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="量子多体动力学数据看板", layout="wide")

# ================= 核心配置区 =================
# 【修改点】使用自适应相对路径。云端和本地通用！
# 它会自动寻找和 app.py 放在同一个文件夹下的 "Data" 文件夹
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
                    eta_str = file.split('eta')[1].replace('.npz', '')
                    eta = float(eta_str)
                except Exception:
                    continue

                file_path = os.path.join(root, file)
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
        df = df.sort_values(by=['L', 'Init', 'Freq', 'U', 'eta']).reset_index(drop=True)
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
    """根据传递误差对时间序列进行截断"""
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

# ================= 3. 左侧栏：全局设置与单组探索 =================
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

if df_f.empty:
    st.error("该实验条件下暂时没有数据，请调整参数。")
    st.stop()

current_data_row = df_f.iloc[0].to_dict()

# ================= 4. 顶部：信息板 =================
st.title("量子多体动力学看板")
col_algo, col_chi, col_nmax, col_bc = st.columns(4)
col_algo.metric("算法", "TEBD (TeNPy)")
col_chi.metric("截断维数 (χ)", current_data_row['chi'])
col_nmax.metric("局域玻色子 (n_max)", current_data_row['nmax'])
col_bc.metric("边界条件", current_data_row['bc'])
st.markdown("---")

# ================= 5. 主展区 =================
col_main, col_side = st.columns([3, 1])

with col_main:
    tab_single, tab_compare = st.tabs(["单组数据探索", "跨条件自由构建对比"])

    # ---------------- 标签页 A：单组数据 ----------------
    with tab_single:
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
            target_range = col_cfg1.selectbox("中心局域范围 (包含格点数)", range_opts, index=len(range_opts) - 1)

            metric_opts = [
                "N全平均", "N_odd平均", "N_even平均", "Imbalance",
                "P0全平均", "P0_odd平均", "P0_even平均",
                "P1全平均", "P1_odd平均", "P1_even平均",
                "P2全平均", "P2_odd平均", "P2_even平均"
            ]
            target_metrics = col_cfg2.multiselect("输出物理量", metric_opts, default=["N全平均"])
            config_val = target_range
            label_prefix = f"范围 {target_range}"

        st.markdown("---")

        # 截断设置
        c_trunc1, c_trunc2 = st.columns([1, 2])
        single_cutoff_mode = c_trunc1.radio("误差截断模式", ["无截断 (全长)", "自定义误差截断"], horizontal=True)
        single_err_limit = c_trunc2.number_input("传递误差阈值 (%)", value=2.0,
                                                 step=0.5) if single_cutoff_mode == "自定义误差截断" else None

        # 加载真实数据
        times, occ_arr, P0_arr, P1_arr, P2_arr, err_prop = get_real_data(current_data_row['file_path'], time_axis_unit)

        if len(times) > 0 and target_metrics:
            fig, ax = plt.subplots(figsize=(9, 4))
            for metric in target_metrics:
                y_data = process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, param_L, obs_mode_global,
                                             config_val, metric)
                # 应用误差截断
                t_plot, y_plot = apply_truncation(times, y_data, err_prop, single_cutoff_mode, single_err_limit)
                ax.plot(t_plot, y_plot, label=f"{label_prefix} - {metric}")

            ax.set_xlabel(f"Time ({time_axis_unit})")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()
            st.pyplot(fig)

            # 【新增】传递误差展示区
            with st.expander("查看传递误差累积曲线 (Propagation Error)", expanded=False):
                fig_err, ax_err = plt.subplots(figsize=(9, 2.5))
                # 注意：这里展示的是未截断的原始全长误差数据，以便用户观察全局
                ax_err.plot(times, err_prop, color='orange', linestyle='-', label='传递误差 (Propagation Error)')

                # 如果开启了自定义截断，画一条红线指示阈值位置
                if single_cutoff_mode == "自定义误差截断" and single_err_limit is not None:
                    ax_err.axhline(single_err_limit, color='r', linestyle='--', label=f'截断阈值: {single_err_limit}%')

                ax_err.set_xlabel(f"Time ({time_axis_unit})")
                ax_err.set_ylabel("Error (%)")
                ax_err.grid(True, alpha=0.3)
                ax_err.legend()
                st.pyplot(fig_err)

                # 严谨的学术提示
                st.info(
                    "ℹ️ 注：当前图表仅展示 **传递误差 (Propagation Error)**。截断误差 (Truncation Error) 等其他演化误差的影响，请参阅右侧面板的『截断收敛性验证 (χ)』静态报告。")

        elif len(times) == 0:
            st.error("数据读取失败，请检查 npz 文件。")
        else:
            st.info("请选择至少一个物理量。")

    # ---------------- 标签页 B：跨条件四级搜索构建器 ----------------
    with tab_compare:
        st.markdown("#### 自由对比曲线构建器 (4级选取)")

        # 第一步：物理条件融合为一个下拉菜单，避免过多组件拥挤
        all_exp_strs = []
        for _, row in df_registry.iterrows():
            all_exp_strs.append(
                f"L={row['L']} | {row['Init']} | F={row['Freq']} | U={row['U']} | J={row['J']} | η={row['eta']}")

        b_exp_str = st.selectbox("[1] 选择基础物理条件 (尺寸/初态/频率/U/J/η)", all_exp_strs)
        # 反向解析选中行的 L 值
        b_idx = all_exp_strs.index(b_exp_str)
        b_row_data = df_registry.iloc[b_idx].to_dict()
        b_L = b_row_data['L']

        c4, c5, c6 = st.columns(3)
        b_obs_mode = c4.radio("[2] 观察区域", ["单格点", "局域范围"], horizontal=True, key="b_obs_mode")

        if b_obs_mode == "单格点":
            b_site_or_range = c5.number_input("[3] 格点数 (1~L)", min_value=1, max_value=b_L, value=b_L // 2 + 1,
                                              key="b_site")
            b_metric_opts = ["和现有输出一致", "N", "P0", "P1", "P2"]
            prefix_desc = f"格点 {b_site_or_range}"
        else:
            b_range_opts = [i for i in range(1, b_L + 1) if i % 2 != 0]
            if b_L not in b_range_opts: b_range_opts.append(b_L)
            b_site_or_range = c5.selectbox("[3] 局域范围", b_range_opts, index=len(b_range_opts) - 1, key="b_range")
            b_metric_opts = [
                "和现有输出一致",
                "N全平均", "N_odd平均", "N_even平均", "Imbalance",
                "P0全平均", "P0_odd平均", "P0_even平均",
                "P1全平均", "P1_odd平均", "P1_even平均",
                "P2全平均", "P2_odd平均", "P2_even平均"
            ]
            prefix_desc = f"范围 {b_site_or_range}"

        b_metric = c6.selectbox("[4] 物理量", b_metric_opts, key="b_metric")

        if st.button("将该曲线加入对比池"):
            metrics_to_add = []
            if b_metric == "和现有输出一致":
                if b_obs_mode != obs_mode_global:
                    st.error("操作被拒绝：Tab 2 选择的【观察区域】与 Tab 1 不一致，无法直接克隆。")
                elif not target_metrics:
                    st.warning("Tab 1 中没有选择任何物理量。")
                else:
                    metrics_to_add = target_metrics
            else:
                metrics_to_add = [b_metric]

            for m in metrics_to_add:
                line_config = {
                    'desc': f"{b_exp_str.replace(' | ', ', ')} | {prefix_desc} {m}",
                    'file_path': b_row_data['file_path'],
                    'L': b_L, 'obs_mode': b_obs_mode,
                    'site_or_range': b_site_or_range, 'metric': m
                }
                if line_config not in st.session_state.compare_lines:
                    st.session_state.compare_lines.append(line_config)
                    st.success(f"已加入: {line_config['desc']}")

        st.markdown("---")

        if st.session_state.compare_lines:
            c_align1, c_align2 = st.columns([1, 2])
            comp_x_mode = c_align1.radio("横轴对齐基准", ["以最大长度为准", "以最短长度为准", "自定义误差截断"],
                                         horizontal=True)
            comp_err_limit = c_align2.number_input("对比时传递误差截断 (%)", value=2.0,
                                                   step=0.5) if comp_x_mode == "自定义误差截断" else None

            if st.button("清空对比池"):
                st.session_state.compare_lines = []
                st.rerun()

            fig_c, ax_c = plt.subplots(figsize=(9, 4.5))
            max_times = []

            for line in st.session_state.compare_lines:
                t_m, o_m, p0_m, p1_m, p2_m, e_m = get_real_data(line['file_path'], time_axis_unit)
                y_m = process_target_data(t_m, o_m, p0_m, p1_m, p2_m, line['L'], line['obs_mode'],
                                          line['site_or_range'], line['metric'])

                # 在画图前应用截断
                if comp_x_mode == "自定义误差截断":
                    t_plot, y_plot = apply_truncation(t_m, y_m, e_m, comp_x_mode, comp_err_limit)
                else:
                    t_plot, y_plot = t_m, y_m

                if len(t_plot) > 0:
                    max_times.append(t_plot[-1])
                    ax_c.plot(t_plot, y_plot, label=line['desc'])

            # 处理长短轴对齐
            if len(max_times) > 0:
                if comp_x_mode == "以最短长度为准":
                    ax_c.set_xlim(0, min(max_times))
                elif comp_x_mode == "以最大长度为准":
                    ax_c.set_xlim(0, max(max_times))

            ax_c.set_xlabel(f"Time ({time_axis_unit})")
            ax_c.set_ylabel("Value")
            ax_c.legend(loc='upper right', bbox_to_anchor=(1.0, 1.15))
            ax_c.grid(True, alpha=0.3)
            st.pyplot(fig_c)
        else:
            st.caption("对比池为空，请在上方构建曲线。")

# ---------------- 6. 右侧：静态收敛性展示 ----------------
with col_side:
    st.markdown("### 截断收敛性验证")
    st.markdown(f"**当前选中系统 χ = {current_data_row['chi']}**")

    with st.expander(f"查看针对 χ={current_data_row['chi']} 的收敛性测试报告", expanded=True):
        st.markdown("正在统计...")