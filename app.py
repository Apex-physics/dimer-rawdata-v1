import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import io
import datetime
import re  # 用于极简的正则提取

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


# ================= 通用下载组件生成器 =================
def render_download_buttons(fig, df, file_prefix):
    """为图表生成 PNG 和 CSV 的下载按钮"""
    # 保存图片到内存
    buf_img = io.BytesIO()
    fig.savefig(buf_img, format="png", bbox_inches="tight", dpi=300)
    buf_img.seek(0)

    # 保存 CSV 到内存 (utf-8-sig 防止 Excel 乱码)
    csv_data = df.to_csv(index=False).encode('utf-8-sig')

    c1, c2 = st.columns(2)
    c1.download_button(
        label="🖼️ 下载图表 (PNG)",
        data=buf_img,
        file_name=f"{file_prefix}.png",
        mime="image/png",
        key=f"png_{file_prefix}"
    )
    c2.download_button(
        label="📊 下载数据 (CSV)",
        data=csv_data,
        file_name=f"{file_prefix}.csv",
        mime="text/csv",
        key=f"csv_{file_prefix}"
    )


# ================= 1. 数据扫描与建库逻辑 =================
def scan_and_build_registry():
    records = []
    if not os.path.exists(DATA_DIR):
        return False, f"找不到数据目录: {DATA_DIR}"

    for root, dirs, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        if not folder_name.startswith("L="):
            continue

        try:
            folder_params = dict(item.split('=') for item in folder_name.split('_'))
            L_val = int(folder_params.get('L', 0))
            init_val = folder_params.get('Init', 'Unknown')
            freq_val = folder_params.get('Freq', 'Unknown')
            U_val = float(folder_params.get('U', 0.0))
            J_val = float(folder_params.get('J', 1.0))
        except Exception:
            continue

        for file in files:
            if file.endswith('.npz') and file.startswith('SimData'):
                try:
                    eta = 0.0
                    eta_match = re.search(r'eta[=]?([\d\.]+)', file)
                    if eta_match:
                        eta = float(eta_match.group(1))

                    chi = 512
                    chi_match = re.search(r'chi[=]?(\d+)', file)
                    if chi_match:
                        chi = int(chi_match.group(1))

                    file_path = os.path.join(root, file)
                    data = np.load(file_path, allow_pickle=True)
                    if "chi" not in file:
                        chi = int(data['metadata'][0].get('chi_max', 512))

                    n_max = int(data['metadata'][0].get('n_max', 3))
                    data.close()

                    records.append({
                        'L': L_val, 'Init': init_val, 'Freq': freq_val, 'U': U_val, 'J': J_val,
                        'eta': eta, 'chi': chi, 'nmax': n_max, 'bc': 'OBC', 'file_path': file_path
                    })
                except Exception:
                    continue

    if records:
        df = pd.DataFrame(records)
        df = df.sort_values(by=['L', 'Init', 'Freq', 'U', 'eta', 'chi']).reset_index(drop=True)
        df.to_csv(REGISTRY_FILE, index=False)
        return True, len(df)
    return False, "未找到有效数据"


@st.cache_data
def load_registry(refresh_tag):
    if os.path.exists(REGISTRY_FILE):
        return pd.read_csv(REGISTRY_FILE)
    return pd.DataFrame()


# ================= 2. 数据读取与计算引擎 =================
@st.cache_data
def get_real_data(file_path, time_unit='t*J'):
    try:
        data = np.load(file_path, allow_pickle=True)
        times = data['times_tJ'] if time_unit == 't*J' else data['times_ms']
        res = (times,
               data['occ_arr'] if 'occ_arr' in data else None,
               data['P0_arr'] if 'P0_arr' in data else None,
               data['P1_arr'] if 'P1_arr' in data else None,
               data['P2_arr'] if 'P2_arr' in data else None,
               data['err_prop'] if 'err_prop' in data else np.zeros_like(times))
        data.close()
        return res
    except:
        return np.array([]), None, None, None, None, np.array([])


def process_target_data(times, occ_arr, P0_arr, P1_arr, P2_arr, L, obs_mode, site_or_range, metric):
    if len(times) == 0: return None
    center_idx = L // 2
    arr_map = {"N": occ_arr, "P0": P0_arr, "P1": P1_arr, "P2": P2_arr}
    base_m = metric.split('平均')[0].split('_')[0] if '平均' in metric else metric
    target = arr_map.get(base_m)
    if target is None: return None

    if obs_mode == "单格点":
        return target[:, site_or_range - 1]
    else:
        start, end = max(0, center_idx - site_or_range // 2), min(L, center_idx + site_or_range // 2 + 1)
        region = target[:, start:end]
        mask = np.array([(start + i) % 2 != 0 for i in range(end - start)])
        if "全平均" in metric: return np.mean(region, axis=1)
        if "_odd平均" in metric: return np.mean(region[:, mask], axis=1)
        if "_even平均" in metric: return np.mean(region[:, ~mask], axis=1)
        if metric == "Imbalance":
            no, ne = np.mean(region[:, mask], axis=1), np.mean(region[:, ~mask], axis=1)
            return (no - ne) / (no + ne + 1e-9)
    return None


def apply_truncation(times, y_data, error, cutoff_mode, limit=None):
    if cutoff_mode == "自定义误差截断" and limit is not None:
        idx = np.where(error > limit)[0]
        if len(idx) > 0: return times[:idx[0]], y_data[:idx[0]]
    return times, y_data


# ================= 3. 主界面布局 =================
if 'compare_lines' not in st.session_state: st.session_state.compare_lines = []
if 'last_refresh' not in st.session_state: st.session_state['last_refresh'] = "从未刷新"

st.sidebar.markdown("### 数据维护")
if st.sidebar.button("扫描硬盘刷新数据"):
    with st.spinner("同步数据中..."):
        s, m = scan_and_build_registry()
        if s: st.session_state['last_refresh'] = datetime.datetime.now().strftime("%H:%M:%S")

df_registry = load_registry(st.session_state['last_refresh'])
if df_registry.empty: st.error("数据字典为空，请点击上方按钮刷新"); st.stop()

time_unit = st.sidebar.radio("时间单位", ["t*J", "ms"])
obs_mode_global = st.sidebar.radio("观察模式", ["单格点", "局域范围"])

# --- 单组筛选逻辑 (隐藏 chi=700) ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 物理参数筛选")
p_L = st.sidebar.selectbox("L", sorted(df_registry['L'].unique()))
df_f = df_registry[df_registry['L'] == p_L]
p_init = st.sidebar.selectbox("Init", sorted(df_f['Init'].unique()))
df_f = df_f[df_f['Init'] == p_init]
p_freq = st.sidebar.selectbox("Freq", sorted(df_f['Freq'].unique()))
df_f = df_f[df_f['Freq'] == p_freq]
p_U = st.sidebar.selectbox("U (Hz)", sorted(df_f['U'].unique()))
df_f = df_f[df_f['U'] == p_U]
p_J = st.sidebar.selectbox("J (Hz)", sorted(df_f['J'].unique()))
df_f = df_f[df_f['J'] == p_J]
p_eta = st.sidebar.selectbox("η", sorted(df_f['eta'].unique()))
df_f = df_f[df_f['eta'] == p_eta]

available_chis_for_info = sorted(df_f['chi'].unique(), reverse=True)
valid_chis = [c for c in available_chis_for_info if c != 700]

if not valid_chis: st.error("无可用生产数据点"); st.stop()
current_row = df_f[df_f['chi'] == valid_chis[0]].iloc[0].to_dict()

# ================= 4. 顶部：信息板 =================
st.title("量子多体动力学看板")
col_algo, col_chi, col_nmax, col_bc = st.columns(4)
col_algo.metric("算法", "TEBD (TeNPy)")

# 使用 HTML 渲染 chi 列表，放大字号防止遮断
chi_str = ", ".join(map(str, available_chis_for_info))
col_chi.markdown(
    f"<div style='margin-top: 0px;'>"
    f"<span style='font-size:0.9rem; color:gray;'>包含的 χ (此组合下)</span><br>"
    f"<span style='font-size:1.8rem; font-weight:bold; color:var(--text-color);'>{chi_str}</span>"
    f"</div>",
    unsafe_allow_html=True
)

col_nmax.metric("n_max", current_row['nmax'])
col_bc.metric("最后扫描", st.session_state['last_refresh'])
st.markdown("---")

# ================= 5. 主展区 (Tabs 布局分离) =================
tab_single, tab_compare = st.tabs(["📊 单组数据探索", "🔍 批量对比(多组)"])

# ----------------- Tab 1：单组数据 -----------------
with tab_single:
    # 构建内部 3:1 排版，保证误差和收敛验证仅在此页面展示
    c_single_main, c_single_side = st.columns([3, 1])

    with c_single_main:
        cfg1, cfg2 = st.columns(2)
        if obs_mode_global == "单格点":
            t_site = cfg1.number_input("格点索引", 1, p_L, p_L // 2 + 1)
            t_metrics = cfg2.multiselect("物理量", ["N", "P0", "P1", "P2"], default=["N"])
            conf_v = t_site
        else:
            r_opts = [i for i in range(1, p_L + 1) if i % 2 != 0]
            if p_L not in r_opts: r_opts.append(p_L)
            t_range = cfg1.selectbox("局域范围", r_opts, index=len(r_opts) - 1)
            m_opts = ["N全平均", "N_odd平均", "N_even平均", "Imbalance", "P2全平均"]
            t_metrics = cfg2.multiselect("物理量", m_opts, default=["N全平均"])
            conf_v = t_range

        st.markdown("---")
        t_cur, o_cur, p0_cur, p1_cur, p2_cur, e_cur = get_real_data(current_row['file_path'], time_unit)

        # --- 单组：主图绘制 ---
        if len(t_cur) > 0 and t_metrics:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            df_main_csv = pd.DataFrame({'Time': t_cur})

            for m in t_metrics:
                y = process_target_data(t_cur, o_cur, p0_cur, p1_cur, p2_cur, p_L, obs_mode_global, conf_v, m)
                if y is not None:
                    ax.plot(t_cur, y, label=m)
                    df_main_csv[m] = y

            ax.set_xlabel(f"Time ({time_unit})")
            # 【图例】置于底部，多列对齐，防遮挡
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False,
                      ncol=max(1, len(t_metrics)))
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            render_download_buttons(fig, df_main_csv, "Single_MainPlot")

            # --- 单组：误差监控 (默认折叠) ---
            with st.expander("📉 展开查看：传递误差累积曲线 (Propagation Error)", expanded=False):
                fig_e, ax_e = plt.subplots(figsize=(9, 2.5))
                ax_e.plot(t_cur, e_cur, color='orange', label='传递误差')
                ax_e.set_xlabel(f"Time ({time_unit})")
                ax_e.set_ylabel("Error (%)")
                # 【图例】置于底部
                ax_e.legend(loc='upper center', bbox_to_anchor=(0.5, -0.25), ncol=2)
                ax_e.grid(True, alpha=0.2)
                st.pyplot(fig_e)
                df_error_csv = pd.DataFrame({'Time': t_cur, 'Error(%)': e_cur})
                render_download_buttons(fig_e, df_error_csv, "Single_ErrorPlot")

    with c_single_side:
        # --- 单组侧栏：收敛性验证 (默认折叠) ---
        with st.expander("💠 展开：截断收敛性验证", expanded=False):
            ref_q = df_registry[
                (df_registry['L'] == p_L) & (df_registry['Init'] == p_init) & (df_registry['chi'] == 700)]
            if ref_q.empty:
                st.info("暂无 χ=700 基准数据")
            else:
                row700 = ref_q.iloc[0]
                match_q = df_registry[(df_registry['L'] == p_L) & (df_registry['Init'] == p_init) &
                                      (df_registry['Freq'] == row700['Freq']) & (df_registry['U'] == row700['U']) &
                                      (df_registry['eta'] == row700['eta']) & (df_registry['chi'] != 700)]
                if match_q.empty:
                    st.info("无同条件对比数据")
                else:
                    row_c = match_q.sort_values('chi', ascending=False).iloc[0]
                    test_m = t_metrics[-1] if t_metrics else "N_even平均"

                    tr, or_, pr0, pr1, pr2, _ = get_real_data(row700['file_path'], time_unit)
                    yr = process_target_data(tr, or_, pr0, pr1, pr2, p_L, obs_mode_global, conf_v, test_m)

                    tc, oc, pc0, pc1, pc2, _ = get_real_data(row_c['file_path'], time_unit)
                    yc = process_target_data(tc, oc, pc0, pc1, pc2, p_L, obs_mode_global, conf_v, test_m)

                    if yr is not None and yc is not None:
                        fig_cv, ax_cv = plt.subplots(figsize=(4, 3.5))
                        ax_cv.plot(tr, yr, 'k--', label="χ=700 (基准)", alpha=0.6)
                        ax_cv.plot(tc, yc, 'r-', label=f"χ={row_c['chi']}")
                        ax_cv.set_title(f"对比量: {test_m}", fontsize=10)
                        # 【图例】底部对齐
                        ax_cv.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=8)
                        st.pyplot(fig_cv)

                        # 误差判定与对齐输出
                        mlen = min(len(yr), len(yc))
                        df_cv_csv = pd.DataFrame({
                            'Time': tc[:mlen],
                            'chi_700': yr[:mlen],
                            f'chi_{row_c["chi"]}': yc[:mlen]
                        })
                        render_download_buttons(fig_cv, df_cv_csv, "Single_Convergence")

                        diff = np.abs(yr[:mlen] - yc[:mlen])
                        norm = np.max(np.abs(yr)) if np.max(np.abs(yr)) > 1e-9 else 1.0
                        fail_idx = np.where(diff / norm >= 0.01)[0]
                        st.markdown("---")
                        if len(fail_idx) > 0:
                            st.write(f"判定：在 **{tc[fail_idx[0]]:.2f} {time_unit}** 时，相对误差达到 1%")
                        else:
                            st.write("判定：在全部演化时间内，相对误差未达到 1%")

# ----------------- Tab 2：批量对比 (多组) -----------------
with tab_compare:
    st.markdown("#### 级联选择器 (从左至右顺序选择)")
    df_c = df_registry.copy()

    # 强制级联多选 (不含 chi)
    cl1, cl2, cl3, cl4, cl5, cl6 = st.columns(6)
    s_L = cl1.multiselect("L", sorted(df_c['L'].unique()))
    df_c = df_c[df_c['L'].isin(s_L)] if s_L else df_c.iloc[0:0]
    s_In = cl2.multiselect("Init", sorted(df_c['Init'].unique()))
    df_c = df_c[df_c['Init'].isin(s_In)] if s_In else df_c.iloc[0:0]
    s_Fr = cl3.multiselect("Freq", sorted(df_c['Freq'].unique()))
    df_c = df_c[df_c['Freq'].isin(s_Fr)] if s_Fr else df_c.iloc[0:0]
    s_U = cl4.multiselect("U", sorted(df_c['U'].unique()))
    df_c = df_c[df_c['U'].isin(s_U)] if s_U else df_c.iloc[0:0]
    s_J = cl5.multiselect("J", sorted(df_c['J'].unique()))
    df_c = df_c[df_c['J'].isin(s_J)] if s_J else df_c.iloc[0:0]
    s_Et = cl6.multiselect("η", sorted(df_c['eta'].unique()))
    df_c = df_c[df_c['eta'].isin(s_Et)] if s_Et else df_c.iloc[0:0]

    st.markdown("---")
    co1, co2, co3 = st.columns(3)
    b_obs = co1.radio("模式", ["单格点", "局域范围"], key="b_obs")
    uLs = sorted(list(set(s_L)))

    if b_obs == "单格点":
        limL = min(uLs) if uLs else 1
        b_val = co2.number_input(f"格点 (最大限制 {limL})", 1, limL, limL // 2 + 1)
    else:
        if len(uLs) > 1:
            b_val = co2.selectbox("动态范围", ["全局", "L-2", "L-4", "中心单点"])
        else:
            lv = uLs[0] if uLs else 1
            opts = [i for i in range(1, lv + 1) if i % 2 != 0]
            b_val = co2.selectbox("范围", opts, index=len(opts) - 1)

    b_m = co3.selectbox("对比物理量", ["和现有输出一致", "N全平均", "Imbalance", "P2全平均", "N"])

    if st.button("将选中组合加入对比池"):
        if df_c.empty:
            st.warning("当前筛选下无数据，请完整勾选上方参数。")
        else:
            metrics = [b_m] if b_m != "和现有输出一致" else t_metrics
            for _, row in df_c.iterrows():
                ar = b_val
                if isinstance(ar, str):
                    if ar == "全局":
                        ar = row['L']
                    elif ar == "中心单点":
                        ar = 1
                    else:
                        ar = max(1, row['L'] - int(ar.split('-')[1]))
                for m in metrics:
                    st.session_state.compare_lines.append({
                        'desc': f"L={row['L']}|η={row['eta']}|χ={row['chi']}|{m}",
                        'file_path': row['file_path'], 'L': row['L'], 'obs_mode': b_obs, 'site_or_range': ar,
                        'metric': m
                    })
            st.rerun()

    if st.session_state.compare_lines:
        if st.button("清空对比池"):
            st.session_state.compare_lines = []
            st.rerun()

        fig_c, ax_c = plt.subplots(figsize=(10, 4.5))

        # 组装下载用长格式 CSV 数据
        multi_csv_rows = []

        for line in st.session_state.compare_lines:
            tb, ob, pb0, pb1, pb2, _ = get_real_data(line['file_path'], time_unit)
            yb = process_target_data(tb, ob, pb0, pb1, pb2, line['L'], line['obs_mode'], line['site_or_range'],
                                     line['metric'])
            if yb is not None:
                ax_c.plot(tb, yb, label=line['desc'])
                # 将该条曲线加入批量记录列表
                for t_val, y_val in zip(tb, yb):
                    multi_csv_rows.append({'Legend': line['desc'], 'Time': t_val, 'Value': y_val})

        ax_c.set_xlabel(f"Time ({time_unit})")
        # 【图例】置于画布的正右侧（不遮挡主图）
        ax_c.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), fontsize='small', title="曲线说明")
        ax_c.grid(True, alpha=0.3)
        st.pyplot(fig_c)

        df_multi_csv = pd.DataFrame(multi_csv_rows)
        render_download_buttons(fig_c, df_multi_csv, "MultiCompare_Plot")