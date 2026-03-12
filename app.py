import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import datetime
import re  # 仅用于处理文件名中 eta=0.2 和 eta0.2 这两种差异

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


# ================= 导出辅助函数 (不修改原有逻辑) =================
def export_plot_and_csv(fig, df, base_filename):
    """提供图表和数据的下载功能"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=200)
    st.download_button(label="下载图表 (PNG)", data=buf.getvalue(), file_name=f"{base_filename}.png", mime="image/png")

    csv = df.to_csv(index=False).encode('utf-8-sig')
    st.download_button(label="下载数据 (CSV)", data=csv, file_name=f"{base_filename}.csv", mime="text/csv")


# ================= 1. 数据扫描与建库逻辑 =================
def scan_and_build_registry():
    records = []
    if not os.path.exists(DATA_DIR):
        return False, f"找不到数据目录: {DATA_DIR}"

    for root, dirs, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)

        # 【恢复为您原始完美的文件夹解析逻辑】
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

        # 遍历内部 npz 文件
        for file in files:
            if file.endswith('.npz') and file.startswith('SimData'):
                try:
                    # 【极简修改】智能提取 eta，兼容 "eta=0.200" 和 "eta0.200" 两种格式
                    eta = 0.0
                    eta_match = re.search(r'eta[=]?([\d\.]+)', file)
                    if eta_match:
                        eta = float(eta_match.group(1))

                    file_path = os.path.join(root, file)
                    data = np.load(file_path, allow_pickle=True)
                    meta_dict = data['metadata'][0]

                    # 智能提取 chi，兼容 "chi=700" 和 "chi700"
                    chi_match = re.search(r'chi[=]?(\d+)', file)
                    if chi_match:
                        chi = int(chi_match.group(1))
                    else:
                        chi = int(meta_dict.get('chi_max', 512))

                    nmax = int(meta_dict.get('n_max', 3))
                    bc = 'OBC'
                    data.close()

                    records.append({
                        'L': L, 'Init': init_state, 'Freq': freq, 'U': U, 'J': J,
                        'eta': eta, 'chi': chi, 'nmax': nmax, 'bc': bc, 'file_path': file_path
                    })
                except Exception:
                    continue

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
    """读取真实的 npz 数据包，兼容缺失某些物理量的情况"""
    try:
        data = np.load(file_path, allow_pickle=True)
        times = data['times_tJ'] if time_unit == 't*J' else data['times_ms']
        occ_arr = data['occ_arr'] if 'occ_arr' in data else None
        P0_arr = data['P0_arr'] if 'P0_arr' in data else None
        P1_arr = data['P1_arr'] if 'P1_arr' in data else None
        P2_arr = data['P2_arr'] if 'P2_arr' in data else None
        err_prop = data['err_prop'] if 'err_prop' in data else np.zeros_like(times)
        data.close()
        return times, occ_arr, P0_arr, P1_arr, P2_arr, err_prop
    except Exception as e:
        return np.array([]), None, None, None, None, np.array([])


def process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, L, obs_mode, site_or_range, metric):
    """根据选项切片并计算物理量，数据缺失时返回 None"""
    if len(times) == 0: return None
    center_idx = L // 2

    arr_map = {
        "N": occ_arr, "P0": P0_arr, "P1": P1_arr, "P2": P2_arr,
        "N全平均": occ_arr, "N_odd平均": occ_arr, "N_even平均": occ_arr, "Imbalance": occ_arr,
        "P0全平均": P0_arr, "P0_odd平均": P0_arr, "P0_even平均": P0_arr,
        "P1全平均": P1_arr, "P1_odd平均": P1_arr, "P1_even平均": P1_arr,
        "P2全平均": P2_arr, "P2_odd平均": P2_arr, "P2_even平均": P2_arr
    }

    target_arr = arr_map.get(metric)
    if target_arr is None: return None

    if obs_mode == "单格点":
        idx = site_or_range - 1
        return target_arr[:, idx]

    elif obs_mode == "局域范围":
        R = site_or_range
        start_idx = max(0, center_idx - R // 2)
        end_idx = min(L, center_idx + R // 2 + 1)
        region_arr = target_arr[:, start_idx:end_idx]

        odd_mask = np.array([(start_idx + i) % 2 != 0 for i in range(end_idx - start_idx)])
        even_mask = ~odd_mask

        if "全平均" in metric: return np.mean(region_arr, axis=1)
        if "_odd平均" in metric: return np.mean(region_arr[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(
            len(times))
        if "_even平均" in metric: return np.mean(region_arr[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(
            len(times))
        if metric == "Imbalance":
            N_odd_avg = np.mean(region_arr[:, odd_mask], axis=1) if np.any(odd_mask) else np.zeros(len(times))
            N_even_avg = np.mean(region_arr[:, even_mask], axis=1) if np.any(even_mask) else np.zeros(len(times))
            return (N_odd_avg - N_even_avg) / (N_odd_avg + N_even_avg + 1e-9)

    return None


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

# 单数据锁定为最大有效 chi (排除700)
available_chis_for_info = sorted(df_f['chi'].unique(), reverse=True)
valid_chis = [c for c in available_chis_for_info if c != 700]

if not valid_chis:
    st.error("该实验条件下除基准(chi=700)外无生产数据。")
    st.stop()

active_chi = max(valid_chis)
current_data_row = df_f[df_f['chi'] == active_chi].iloc[0].to_dict()

# ================= 4. 顶部：信息板 =================
st.title("量子多体动力学数据看板")
col_algo, col_chi, col_nmax, col_bc = st.columns(4)
col_algo.metric("算法", "TEBD (TeNPy)")

# 【修改点 2】输出涵盖的所有 chi，并调整字号显示
chi_str = ", ".join(map(str, available_chis_for_info))
col_chi.markdown(f"**包含的 χ (此组合下)**: <h2 style='margin-top:0'>{chi_str}</h2>", unsafe_allow_html=True)

col_nmax.metric("局域玻色子 (n_max)", current_data_row['nmax'])
col_bc.metric("边界条件", current_data_row['bc'])
st.markdown("---")

# ================= 5. 主展区 =================
col_main, col_side = st.columns([3, 1])

with col_main:
    tab_single, tab_compare = st.tabs(["单组数据探索", "批量自由对比构建器"])

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
            fig, ax = plt.subplots(figsize=(9, 4.5))
            df_main_data = pd.DataFrame({'Time': times})
            for metric in target_metrics:
                y_data = process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, param_L, obs_mode_global,
                                             config_val, metric)
                if y_data is None:
                    st.warning(f"数据包中缺失物理量 [{metric}]")
                    continue
                t_plot, y_plot = apply_truncation(times, y_data, err_prop, single_cutoff_mode, single_err_limit)
                ax.plot(t_plot, y_plot, label=f"{label_prefix} - {metric}")
                df_main_data[metric] = y_data

            ax.set_xlabel(f"Time ({time_axis_unit})")
            ax.set_ylabel("Value")
            # 【修改点 3】图例显示在下方，多列布局防遮挡
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(target_metrics)), fancybox=True,
                      shadow=True)
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            # 【修改点 4】下载主图
            export_plot_and_csv(fig, df_main_data, "Single_Main_Plot")

            # 【修改点 1】误差图折叠展开
            with st.expander("📉 展开查看：传递误差累积曲线 (Propagation Error)", expanded=False):
                fig_err, ax_err = plt.subplots(figsize=(9, 2.5))
                ax_err.plot(times, err_prop, color='orange', linestyle='-', label='传递误差')
                if single_cutoff_mode == "自定义误差截断" and single_err_limit is not None:
                    ax_err.axhline(single_err_limit, color='r', linestyle='--', label=f'截断阈值: {single_err_limit}%')
                ax_err.set_xlabel(f"Time ({time_axis_unit})")
                ax_err.set_ylabel("Error (%)")
                # 【修改点 3】图例移至下方
                ax_err.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
                ax_err.grid(True, alpha=0.3)
                st.pyplot(fig_err)
                # 【修改点 4】下载误差图
                export_plot_and_csv(fig_err, pd.DataFrame({'Time': times, 'Error': err_prop}), "Single_Error_Plot")

        elif len(times) == 0:
            st.error("数据读取失败，请检查 npz 文件。")
        else:
            st.info("请选择至少一个物理量。")

    # ---------------- 标签页 B：批量对比 (重构：删除 chi 筛选) ----------------
    with tab_compare:
        st.markdown("#### 批量自由对比构建器")

        df_c = df_registry.copy()

        # 【修改点 2】删除 col_m7 和 s_chi 筛选按钮，改为 3x2 布局
        col_m1, col_m2, col_m3 = st.columns(3)
        col_m4, col_m5, col_m6 = st.columns(3)

        s_L = col_m1.multiselect("L", sorted(df_c['L'].unique()), default=[])
        df_c = df_c[df_c['L'].isin(s_L)] if s_L else df_c.iloc[0:0]

        s_Init = col_m2.multiselect("Init", sorted(df_c['Init'].unique()), default=[])
        df_c = df_c[df_c['Init'].isin(s_In)] if s_Init else df_c.iloc[0:0]

        s_Freq = col_m3.multiselect("Freq", sorted(df_c['Freq'].unique()), default=[])
        df_c = df_c[df_c['Freq'].isin(s_Freq)] if s_Freq else df_c.iloc[0:0]

        s_U = col_m4.multiselect("U", sorted(df_c['U'].unique()), default=[])
        df_c = df_c[df_c['U'].isin(s_U)] if s_U else df_c.iloc[0:0]

        s_J = col_m5.multiselect("J", sorted(df_c['J'].unique()), default=[])
        df_c = df_c[df_c['J'].isin(s_J)] if s_J else df_c.iloc[0:0]

        s_eta = col_m6.multiselect("η", sorted(df_c['eta'].unique()), default=[])
        df_c = df_c[df_c['eta'].isin(s_eta)] if s_eta else df_c.iloc[0:0]

        st.markdown("---")
        c_obs1, c_obs2, c_obs3 = st.columns(3)
        b_obs_mode = c_obs1.radio("[2] 观察区域", ["单格点", "局域范围"], horizontal=True, key="b_obs_mode")

        unique_Ls = sorted(list(set(s_L)))

        if b_obs_mode == "单格点":
            min_L = min(unique_Ls) if unique_Ls else 1
            b_site_or_range = c_obs2.number_input(f"[3] 格点数 (当前选择组中最小限制 {min_L})", min_value=1,
                                                  max_value=min_L, value=min_L // 2 + 1, key="b_site")
            b_metric_opts = ["和现有输出一致", "N", "P0", "P1", "P2"]
        else:
            if len(unique_Ls) > 1:
                b_range_opts = ["全局", "L-2", "L-4", "中心单点"]
                b_site_or_range = c_obs2.selectbox("[3] 动态局域范围 (适配多L)", b_range_opts, key="b_range")
            else:
                L_val = unique_Ls[0] if unique_Ls else 1
                b_range_opts = [i for i in range(1, L_val + 1) if i % 2 != 0]
                if L_val not in b_range_opts: b_range_opts.append(L_val)
                b_site_or_range = c_obs2.selectbox("[3] 局域范围", b_range_opts, index=len(b_range_opts) - 1,
                                                   key="b_range")

            b_metric_opts = [
                "和现有输出一致", "N全平均", "N_odd平均", "N_even平均", "Imbalance",
                "P0全平均", "P0_odd平均", "P0_even平均", "P1全平均", "P1_odd平均", "P1_even平均",
                "P2全平均", "P2_odd平均", "P2_even平均"
            ]

        b_metric = c_obs3.selectbox("[4] 物理量", b_metric_opts, key="b_metric")

        if st.button("将选中组合批量加入对比池"):
            if df_c.empty:
                st.warning("当前没有选择任何有效数据可添加。")
            else:
                metrics_to_add = [b_metric] if b_metric != "和现有输出一致" else target_metrics
                for _, row in df_c.iterrows():
                    actual_range = b_site_or_range
                    if isinstance(b_site_or_range, str) and b_obs_mode == "局域范围":
                        if b_site_or_range == "全局":
                            actual_range = row['L']
                        elif b_site_or_range == "中心单点":
                            actual_range = 1
                        elif b_site_or_range.startswith("L-"):
                            val = int(b_site_or_range.split("-")[1])
                            actual_range = max(1, row['L'] - val)
                    for m in metrics_to_add:
                        line_config = {
                            'desc': f"L={row['L']}|{row['Init']}|η={row['eta']}|χ={row['chi']}|{m}",
                            'file_path': row['file_path'], 'L': row['L'],
                            'obs_mode': b_obs_mode, 'site_or_range': actual_range, 'metric': m
                        }
                        if line_config not in st.session_state.compare_lines:
                            st.session_state.compare_lines.append(line_config)
                st.success("已添加曲线。")

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

            fig_c, ax_c = plt.subplots(figsize=(10, 4.5))
            df_compare_export = pd.DataFrame()
            for line in st.session_state.compare_lines:
                t_m, o_m, p0_m, p1_m, p2_m, e_m = get_real_data(line['file_path'], time_axis_unit)
                y_m = process_target_data(t_m, o_m, p0_m, p1_m, p2_m, line['L'], line['obs_mode'],
                                          line['site_or_range'], line['metric'])
                if y_m is None: continue
                t_plot, y_plot = apply_truncation(t_m, y_m, e_m, comp_x_mode, comp_err_limit)
                ax_c.plot(t_plot, y_plot, label=line['desc'])
                # 保存对比数据
                df_compare_export[f"{line['desc']}_Time"] = pd.Series(t_plot)
                df_compare_export[f"{line['desc']}_Val"] = pd.Series(y_plot)

            ax_c.set_xlabel(f"Time ({time_axis_unit})")
            # 【修改点 3】批量对比图例：置于右侧，防止遮挡
            ax_c.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small')
            ax_c.grid(True, alpha=0.3)
            st.pyplot(fig_c)
            # 【修改点 4】下载对比图
            export_plot_and_csv(fig_c, df_compare_export, "Batch_Compare_Plot")

# 【修改点 1】收敛性验证移动至“单组数据探索”内的 expander 中实现
with col_side:
    st.markdown("### 截断收敛性验证")
    # 这里我们只保留在 col_main 内部单选 Tab 时的展示逻辑
    with st.expander("💠 展开查看：截断收敛性验证 (χ)", expanded=False):
        ref_df_700 = df_registry[
            (df_registry['L'] == param_L) &
            (df_registry['Init'] == param_init) &
            (df_registry['chi'] == 700)
            ]

        if ref_df_700.empty:
            st.write("暂无 χ=700 对比数据")
        else:
            ref_700_row = ref_df_700.iloc[0]
            cur_df_match = df_registry[
                (df_registry['L'] == param_L) &
                (df_registry['Init'] == param_init) &
                (df_registry['Freq'] == ref_700_row['Freq']) &
                (df_registry['U'] == ref_700_row['U']) &
                (df_registry['J'] == ref_700_row['J']) &
                (df_registry['eta'] == ref_700_row['eta']) &
                (df_registry['chi'] != 700)
                ]

            if cur_df_match.empty:
                st.write("未能找到同条件的普通数据进行对比")
            else:
                cur_match_row = cur_df_match.sort_values(by='chi', ascending=False).iloc[0]
                test_metric = target_metrics[-1] if target_metrics else "N_even平均"

                t_ref, o_ref, p0_ref, p1_ref, p2_ref, _ = get_real_data(ref_700_row['file_path'], time_axis_unit)
                y_ref = process_target_data(t_ref, o_ref, p0_ref, p1_ref, p2_ref, param_L, obs_mode_global, config_val,
                                            test_metric)

                t_cur, o_cur, p0_cur, p1_cur, p2_cur, _ = get_real_data(cur_match_row['file_path'], time_axis_unit)
                y_cur = process_target_data(t_cur, o_cur, p0_cur, p1_cur, p2_cur, param_L, obs_mode_global, config_val,
                                            test_metric)

                if y_ref is not None and y_cur is not None:
                    fig_cv, ax_cv = plt.subplots(figsize=(4, 3.5))
                    ax_cv.plot(t_ref, y_ref, 'k--', label="χ=700", alpha=0.7)
                    ax_cv.plot(t_cur, y_cur, 'r-', label=f"χ={cur_match_row['chi']}", linewidth=1.2)
                    ax_cv.set_title(f"对比量: {test_metric}", fontsize=10)
                    # 【修改点 3】图例移至下方
                    ax_cv.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7)
                    st.pyplot(fig_cv)
                    # 【修改点 4】下载收敛图
                    export_plot_and_csv(fig_cv, pd.DataFrame(
                        {'Time': t_cur[:min(len(y_ref), len(y_cur))], 'Ref': y_ref[:min(len(y_ref), len(y_cur))],
                         'Test': y_cur[:min(len(y_ref), len(y_cur))]}), "Convergence_Plot")

                    mlen = min(len(y_ref), len(y_cur))
                    if mlen > 0:
                        diff = np.abs(y_ref[:mlen] - y_cur[:mlen])
                        norm = np.max(np.abs(y_ref)) if np.max(np.abs(y_ref)) > 1e-9 else 1.0
                        fail_idx = np.where(diff / norm >= 0.01)[0]
                        st.markdown("---")
                        if len(fail_idx) > 0:
                            st.write(f"在 {t_cur[fail_idx[0]]:.2f} {time_axis_unit} 时误差达到 1%")
                        else:
                            st.write("全时间段内误差未达到 1%")
                else:
                    st.write("所选测绘量缺失，无法对比")