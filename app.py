import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
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


# ================= 1. 数据扫描与建库逻辑 (双模式简易版) =================
def scan_and_build_registry():
    records = []
    if not os.path.exists(DATA_DIR):
        return False, f"找不到数据目录: {DATA_DIR}"

    for root, dirs, files in os.walk(DATA_DIR):
        folder_name = os.path.basename(root)
        # 文件夹命名严格一致：L=..._Init=...
        if not folder_name.startswith("L="):
            continue

        try:
            folder_params = dict(item.split('=') for item in folder_name.split('_'))
            L_val = int(folder_params.get('L', 0))
            init_val = folder_params.get('Init', 'Unknown')
            freq_val = folder_params.get('Freq', 'Unknown')
            U_val = float(folder_params.get('U', 0.0))
            J_val = float(folder_params.get('J', 0.0))
        except Exception:
            continue

        for file in files:
            if file.endswith('.npz') and file.startswith('SimData'):
                try:
                    # --- 模式判断：提取 eta ---
                    eta = 0.0
                    if "eta=" in file:
                        # 模式A: SimData_..._eta=0.200_...
                        eta = float(file.split("eta=")[1].split("_")[0].replace(".npz", ""))
                    elif "eta" in file:
                        # 模式B: SimData_L9_eta0.200.npz
                        match = re.search(r"eta([\d\.]+)", file)
                        if match:
                            eta = float(match.group(1))

                    # --- 模式判断：提取 chi ---
                    chi = 512
                    if "chi=" in file:
                        chi = int(file.split("chi=")[1].split("_")[0].replace(".npz", ""))
                    elif "chi" in file:
                        match = re.search(r"chi(\d+)", file)
                        if match:
                            chi = int(match.group(1))

                    file_path = os.path.join(root, file)
                    data = np.load(file_path, allow_pickle=True)
                    # 如果文件名没写chi，从metadata里捞
                    if "chi" not in file:
                        chi = int(data['metadata'][0].get('chi_max', 512))

                    nmax = int(data['metadata'][0].get('n_max', 3))
                    data.close()

                    records.append({
                        'L': L_val, 'Init': init_val, 'Freq': freq_val, 'U': U_val, 'J': J_val,
                        'eta': eta, 'chi': chi, 'nmax': nmax, 'bc': 'OBC', 'file_path': file_path
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


# ================= 2. 真实数据读取引擎 =================
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
    # 处理带后缀的测量量
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


# ================= 3. 主界面逻辑 =================
if 'compare_lines' not in st.session_state: st.session_state.compare_lines = []
if 'last_refresh' not in st.session_state: st.session_state['last_refresh'] = "从未刷新"

st.sidebar.markdown("### 数据管理")
if st.sidebar.button("刷新本地数据库"):
    with st.spinner("扫描 NPZ 文件中..."):
        s, m = scan_and_build_registry()
        if s: st.session_state['last_refresh'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df_registry = load_registry(st.session_state['last_refresh'])
if df_registry.empty: st.error("请先点击刷新数据库"); st.stop()

time_unit = st.sidebar.radio("横轴单位", ["t*J", "ms"])
obs_mode_global = st.sidebar.radio("区域模式", ["单格点", "局域范围"])

# --- 单组筛选 ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 物理参数选择")
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
p_eta = st.sidebar.selectbox("eta", sorted(df_f['eta'].unique()))
df_f = df_f[df_f['eta'] == p_eta]

# 过滤隐藏的 chi=700，自动展示当前最大常规 chi
valid_chis = sorted([c for c in df_f['chi'].unique() if c != 700], reverse=True)
if not valid_chis: st.error("该条件下无有效数据"); st.stop()
current_row = df_f[df_f['chi'] == valid_chis[0]].iloc[0].to_dict()

st.title("量子多体动力学数据看板")
col_info1, col_info2, col_info3 = st.columns(3)
col_info1.metric("包含的 χ", ", ".join(map(str, sorted(df_f['chi'].unique()))))
col_info2.metric("玻色子上限 (n_max)", current_row['nmax'])
col_info3.metric("最后更新", st.session_state['last_refresh'])
st.markdown("---")

col_main, col_side = st.columns([3, 1])

with col_main:
    tab_single, tab_compare = st.tabs(["单组数据探索", "跨条件自由构建对比"])

    with tab_single:
        # 对象设置
        c1, c2 = st.columns(2)
        if obs_mode_global == "单格点":
            t_site = c1.number_input("格点索引", 1, p_L, p_L // 2 + 1)
            t_metrics = c2.multiselect("输出物理量", ["N", "P0", "P1", "P2"], default=["N"])
            conf_v, lbl_pre = t_site, f"格点 {t_site}"
        else:
            r_opts = [i for i in range(1, p_L + 1) if i % 2 != 0]
            if p_L not in r_opts: r_opts.append(p_L)
            t_range = c1.selectbox("局域范围", r_opts, index=len(r_opts) - 1)
            m_opts = ["N全平均", "N_odd平均", "N_even平均", "Imbalance", "P0全平均", "P0_odd平均", "P0_even平均",
                      "P1全平均", "P1_odd平均", "P1_even平均", "P2全平均", "P2_odd平均", "P2_even平均"]
            t_metrics = c2.multiselect("输出物理量", m_opts, default=["N全平均"])
            conf_v, lbl_pre = t_range, f"范围 {t_range}"

        st.markdown("---")
        # 绘图区
        t_cur, o_cur, p0_cur, p1_cur, p2_cur, e_cur = get_real_data(current_row['file_path'], time_unit)
        if len(t_cur) > 0 and t_metrics:
            fig, ax = plt.subplots(figsize=(9, 4))
            for m in t_metrics:
                y = process_target_data(t_cur, o_cur, p0_cur, p1_cur, p2_cur, p_L, obs_mode_global, conf_v, m)
                if y is not None: ax.plot(t_cur, y, label=m)
            ax.set_xlabel(f"Time ({time_unit})");
            ax.legend();
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)

            st.markdown("##### 传递误差累积曲线 (Propagation Error)")
            fig_e, ax_e = plt.subplots(figsize=(9, 2))
            ax_e.plot(t_cur, e_cur, color='orange')
            ax_e.set_ylabel("Error (%)");
            ax_e.grid(True, alpha=0.2)
            st.pyplot(fig_e)

    with tab_compare:
        st.markdown("#### 批量自由对比构建器 (级联筛选)")
        df_c = df_registry.copy()
        cm1, cm2, cm3, cm4, cm5, cm6, cm7 = st.columns(7)
        # 强制级联逻辑
        s_L = cm1.multiselect("L", sorted(df_c['L'].unique()))
        df_c = df_c[df_c['L'].isin(s_L)] if s_L else df_c.iloc[0:0]
        s_Init = cm2.multiselect("Init", sorted(df_c['Init'].unique()))
        df_c = df_c[df_c['Init'].isin(s_Init)] if s_Init else df_c.iloc[0:0]
        s_Freq = cm3.multiselect("Freq", sorted(df_c['Freq'].unique()))
        df_c = df_c[df_c['Freq'].isin(s_Freq)] if s_Freq else df_c.iloc[0:0]
        s_U = cm4.multiselect("U", sorted(df_c['U'].unique()))
        df_c = df_c[df_c['U'].isin(s_U)] if s_U else df_c.iloc[0:0]
        s_J = cm5.multiselect("J", sorted(df_c['J'].unique()))
        df_c = df_c[df_c['J'].isin(s_J)] if s_J else df_c.iloc[0:0]
        s_eta = cm6.multiselect("η", sorted(df_c['eta'].unique()))
        df_c = df_c[df_c['eta'].isin(s_eta)] if s_eta else df_c.iloc[0:0]
        s_chi = cm7.multiselect("χ", sorted(df_c['chi'].unique()))
        df_c = df_c[df_c['chi'].isin(s_chi)] if s_chi else df_c.iloc[0:0]

        st.markdown("---")
        co1, co2, co3 = st.columns(3)
        b_obs_mode = co1.radio("观察模式", ["单格点", "局域范围"], key="comp_obs")
        unique_Ls = sorted(list(set(s_L)))
        if b_obs_mode == "单格点":
            limit_L = min(unique_Ls) if unique_Ls else 1
            b_site_or_range = co2.number_input(f"格点数 (限制 {limit_L})", 1, limit_L, limit_L // 2 + 1)
        else:
            if len(unique_Ls) > 1:
                b_site_or_range = co2.selectbox("动态范围", ["全局", "L-2", "L-4", "中心单点"])
            else:
                l_v = unique_Ls[0] if unique_Ls else 1
                opts = [i for i in range(1, l_v + 1) if i % 2 != 0]
                b_site_or_range = co2.selectbox("局域范围", opts, index=len(opts) - 1)

        b_metric = co3.selectbox("物理量对比", ["和现有输出一致", "N全平均", "Imbalance", "P2全平均", "N"])

        if st.button("批量添加"):
            if df_c.empty:
                st.warning("请完成参数选择")
            else:
                metrics = [b_metric] if b_metric != "和现有输出一致" else t_metrics
                for _, row in df_c.iterrows():
                    actual_r = b_site_or_range
                    if isinstance(actual_r, str):
                        if actual_r == "全局":
                            actual_r = row['L']
                        elif actual_r == "中心单点":
                            actual_r = 1
                        else:
                            actual_r = max(1, row['L'] - int(actual_r.split('-')[1]))
                    for m in metrics:
                        st.session_state.compare_lines.append({
                            'desc': f"L={row['L']}|η={row['eta']}|χ={row['chi']}|{m}",
                            'file_path': row['file_path'], 'L': row['L'],
                            'obs_mode': b_obs_mode, 'site_or_range': actual_r, 'metric': m
                        })
                st.rerun()

        if st.session_state.compare_lines:
            if st.button("清空池"): st.session_state.compare_lines = []; st.rerun()
            fig_c, ax_c = plt.subplots(figsize=(9, 4))
            for line in st.session_state.compare_lines:
                t_b, o_b, p0_b, p1_b, p2_b, _ = get_real_data(line['file_path'], time_unit)
                y_b = process_target_data(t_b, o_b, p0_b, p1_b, p2_b, line['L'], line['obs_mode'],
                                          line['site_or_range'], line['metric'])
                if y_b is not None: ax_c.plot(t_b, y_b, label=line['desc'])
            ax_c.legend(fontsize='x-small');
            st.pyplot(fig_c)

# ---------------- 6. 右侧收敛性 (隐藏 chi=700 对标) ----------------
with col_side:
    st.markdown("### 截断收敛性验证")
    # 后台检索同条件的 chi=700 基准文件
    ref_query = df_registry[(df_registry['L'] == p_L) & (df_registry['Init'] == p_init) & (df_registry['chi'] == 700)]
    if ref_query.empty:
        st.write("当前无 chi=700 对比数据")
    else:
        ref_row = ref_query.iloc[0]
        # 寻找与基准完全匹配物理条件的普通数据
        match_query = df_registry[(df_registry['L'] == p_L) & (df_registry['Init'] == p_init) &
                                  (df_registry['Freq'] == ref_row['Freq']) & (df_registry['U'] == ref_row['U']) &
                                  (df_registry['eta'] == ref_row['eta']) & (df_registry['chi'] != 700)]
        if match_query.empty:
            st.write("无同条件对比组")
        else:
            target_row = match_query.sort_values('chi', ascending=False).iloc[0]
            # 对标物理量：优先用最后选的
            m_comp = t_metrics[-1] if t_metrics else "N_even平均"

            t_r, o_r, p0_r, p1_r, p2_r, _ = get_real_data(ref_row['file_path'], time_unit)
            y_r = process_target_data(t_r, o_r, p0_r, p1_r, p2_r, p_L, obs_mode_global, conf_v, m_comp)

            t_t, o_t, p0_t, p1_t, p2_t, _ = get_real_data(target_row['file_path'], time_unit)
            y_t = process_target_data(t_t, o_t, p0_t, p1_t, p2_t, p_L, obs_mode_global, conf_v, m_comp)

            if y_r is not None and y_t is not None:
                fig_cv, ax_cv = plt.subplots(figsize=(4, 3))
                ax_cv.plot(t_r, y_r, 'k--', label="χ=700 (Ref)", alpha=0.6)
                ax_cv.plot(t_t, y_t, 'r', label=f"χ={target_row['chi']}")
                ax_cv.set_title(f"量: {m_comp}", fontsize=9);
                ax_cv.legend(fontsize=7);
                st.pyplot(fig_cv)

                # 计算 1% 时间
                mlen = min(len(y_r), len(y_t))
                diff = np.abs(y_r[:mlen] - y_t[:mlen])
                norm = np.max(np.abs(y_r)) if np.max(np.abs(y_r)) > 1e-9 else 1.0
                fail_idx = np.where(diff / norm >= 0.01)[0]
                if len(fail_idx) > 0:
                    st.write(f"在 {t_t[fail_idx[0]]:.2f} {time_unit} 时误差达到 1%")
                else:
                    st.write("全时段误差未达 1%")