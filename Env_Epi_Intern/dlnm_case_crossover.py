# ==============================================================================
# Environmental Epidemiology Analysis Pipeline (Python Re-implementation)
# Topic: Distributed Lag Non-linear Models (DLNM) & Case-Crossover Designs
# Dataset: Chicago NMMAPS (1987–2000)
# ==============================================================================

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import BSpline

# 设置绘图样式
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv("chicagoEnvData.csv")
print("P1 temp:", df['temp'].quantile(0.01)) # 确认输出约等于 -15.0
print("P99 temp:", df['temp'].quantile(0.99)) # 确认输出约等于 28.6

# ==============================================================================
# 模块 1: 底层核心引擎 —— 手写 Python 版 DLNM 交叉基 (Cross-Basis)
# ==============================================================================
# 【教学重点】：
# R 语言的 dlnm::crossbasis() 做了三件事：
# 1. 对时间序列构建滑动滞后窗口矩阵 X_lag (N x (L+1))，存放第 t 天过去的暴露 [x_t, x_{t-1}, ..., x_{t-L}]。
# 2. 计算暴露维度基矩阵 B_var (形状基于 x) 和滞后维度基矩阵 B_lag (形状基于 0..L)。
# 3. 对每一天的滞后暴露求张量积并求和，生成 N x (df_var * df_lag) 的交叉基矩阵。

class DLNMCrossBasis:
    """
    Python 版分布滞后非线性模型交叉基 (Distributed Lag Non-linear Cross-Basis)
    """
    def __init__(self, max_lag=21):
        self.max_lag = max_lag
        self.var_knots = None
        self.var_boundary_knots = None
        self.lag_knots = None
        self.lag_basis = None
        self.df_var = None
        self.df_lag = None

    @staticmethod
    def _logknots(max_lag, n_knots=3):
        """复现 R 包 dlnm::logknots，在对数尺度上均匀取内部节点"""
        # 对数间距能保证早期滞后天数（0-3天）节点密集，捕获急性效应；后期滞后天数节点稀疏
        spacing = np.linspace(np.log(1), np.log(max_lag), n_knots + 2)
        knots = np.exp(spacing[1:-1]) - 1
        return knots

    def fit_transform(self, x, var_quantiles=[0.10, 0.75, 0.90], n_lag_knots=3):
        n = len(x)
        # 1. 滞后历史矩阵 X_lag: shape (N, max_lag + 1)
        x_lag = np.full((n, self.max_lag + 1), np.nan)
        for lag in range(self.max_lag + 1):
            if lag == 0:
                x_lag[:, lag] = x
            else:
                x_lag[lag:, lag] = x[:-lag]

        # 2. 确定暴露维度的自然样条节点 (Natural Spline Knots)
        valid_x = x[~np.isnan(x)]
        self.var_knots = np.quantile(valid_x, var_quantiles)
        self.var_boundary_knots = (np.min(valid_x), np.max(valid_x))

        # 3. 计算滞后维度基矩阵 B_lag: shape (max_lag + 1, df_lag)
        # 滞后维度变量为离散向量: [0, 1, ..., max_lag]
        lags = np.arange(self.max_lag + 1)
        self.lag_knots = self._logknots(self.max_lag, n_knots=n_lag_knots)
        lag_knots_str = ", ".join(map(str, self.lag_knots))
        
        # 使用 patsy 的 cr (Natural Cubic Splines) 生成滞后基
        lag_formula = f"cr(lags, knots=[{lag_knots_str}], lower_bound=0, upper_bound={self.max_lag}) - 1"
        self.lag_basis = np.asarray(dmatrix(lag_formula, {"lags": lags}))
        self.df_lag = self.lag_basis.shape[1]

        # 4. 构建二维交叉基矩阵 (Cross-basis Matrix)
        # 对每一个滞后 l，计算其暴露基 B_var(x_{t-l})，再与 B_lag[l, :] 做 Kronecker/张量相乘累加
        cb_cols = []
        var_knots_str = ", ".join(map(str, self.var_knots))
        b_low, b_high = self.var_boundary_knots
        
        # 预先获取暴露基的维数
        sample_basis = dmatrix(
            f"cr(valid_x, knots=[{var_knots_str}], lower_bound={b_low}, upper_bound={b_high}) - 1",
            {"valid_x": valid_x[:10]}
        )
        self.df_var = sample_basis.shape[1]

        # 初始化交叉基矩阵: shape (N, df_var * df_lag)
        W = np.zeros((n, self.df_var * self.df_lag))

        for l in range(self.max_lag + 1):
            x_l = x_lag[:, l]
            # 填补 NaN 避免 patsy 报错，后续再把行置为 NaN
            mask_valid = ~np.isnan(x_l)
            x_filled = np.where(mask_valid, x_l, self.var_boundary_knots[0])
            
            basis_l = np.asarray(dmatrix(
                f"cr(x_filled, knots=[{var_knots_str}], lower_bound={b_low}, upper_bound={b_high}) - 1",
                {"x_filled": x_filled}
            ))
            basis_l[~mask_valid, :] = 0.0

            # 滞后权重广播相乘 (Tensor product across lag l)
            for j in range(self.df_var):
                for k in range(self.df_lag):
                    col_idx = j * self.df_lag + k
                    W[:, col_idx] += basis_l[:, j] * self.lag_basis[l, k]

        # 如果前 max_lag 天存在滞后缺失，则置为 NaN
        W[:self.max_lag, :] = np.nan
        return W

    def predict(self, model, cb_col_names, at_vals, cen_val=None):
        """
        复现 R 包 dlnm::crosspred，计算指定温度和滞后时间下的 RR 及 95% 置信区间
        """
        # 提取模型中交叉基对应的系数 beta 和协方差矩阵 vcov
        params = model.params[cb_col_names].values
        cov = model.cov_params().loc[cb_col_names, cb_col_names].values

        var_knots_str = ", ".join(map(str, self.var_knots))
        b_low, b_high = self.var_boundary_knots
        
        # 1. 计算给定网格温度点 at_vals 的暴露基: shape (M, df_var)
        basis_at = np.asarray(dmatrix(
            f"cr(at_vals, knots=[{var_knots_str}], lower_bound={b_low}, upper_bound={b_high}) - 1",
            {"at_vals": at_vals}
        ))
        
        # 2. 如果提供了参考温度 (Centering Value, MMT)，进行中心化
        if cen_val is not None:
            basis_cen = np.asarray(dmatrix(
                f"cr(np.array([cen_val]), knots=[{var_knots_str}], lower_bound={b_low}, upper_bound={b_high}) - 1",
                {"cen_val": np.array([cen_val])}
            ))
            basis_at = basis_at - basis_cen

        # 3. 总体累积效应计算 (Sum across all lags 0..L)
        # lag_sum: shape (df_lag,)，即各个滞后基在所有天数的和
        lag_basis_sum = self.lag_basis.sum(axis=0)
        
        # 交叉基的累积设计矩阵: shape (M, df_var * df_lag)
        A_cum = np.zeros((len(at_vals), self.df_var * self.df_lag))
        for j in range(self.df_var):
            for k in range(self.df_lag):
                col_idx = j * self.df_lag + k
                A_cum[:, col_idx] = basis_at[:, j] * lag_basis_sum[k]

        # 累积对数相对危险度 (log-RR) 及其标准误 (SE)
        log_rr_cum = A_cum @ params
        se_cum = np.sqrt(np.sum((A_cum @ cov) * A_cum, axis=1))

        # 4. 二维切片与网格 (M, L+1) 矩阵计算
        mat_rr = np.zeros((len(at_vals), self.max_lag + 1))
        for l in range(self.max_lag + 1):
            A_lag_l = np.zeros((len(at_vals), self.df_var * self.df_lag))
            for j in range(self.df_var):
                for k in range(self.df_lag):
                    col_idx = j * self.df_lag + k
                    A_lag_l[:, col_idx] = basis_at[:, j] * self.lag_basis[l, k]
            mat_rr[:, l] = np.exp(A_lag_l @ params)

        results = {
            "predvar": at_vals,
            "allRRfit": np.exp(log_rr_cum),
            "allRRlow": np.exp(log_rr_cum - 1.96 * se_cum),
            "allRRhigh": np.exp(log_rr_cum + 1.96 * se_cum),
            "matRRfit": mat_rr,
            "lags": np.arange(self.max_lag + 1)
        }
        return results


# ==============================================================================
# 模块 2: 数据读取与 EDA 探索性分析
# ==============================================================================
# 加载 Chicago NMMAPS 数据 (可通过 statsmodels 或本地 CSV 加载，此处用模拟结构载入演示)
# 对应 R: data(chicagoNMMAPS); summary(chicagoNMMAPS); miss_var_summary()
print(">>> [Step 2] 正在载入与探索芝加哥 NMMAPS 数据集...")

# 构造符合 NMMAPS 实际分布的测试数据（若本地有 chicagoEnvData.csv 可用 pd.read_csv 读取）
try:
    df = pd.read_csv("chicagoEnvData.csv")
    df['date'] = pd.to_datetime(df['date'])
except FileNotFoundError:
    # 模拟 14 年数据结构
    dates = pd.date_range("1987-01-01", "2000-12-31", freq="D")
    n_days = len(dates)
    np.random.seed(42)
    temp = 12 + 15 * np.sin(2 * np.pi * np.arange(n_days) / 365.25 - 1.5) + np.random.normal(0, 4, n_days)
    pm10 = np.random.gamma(shape=3, scale=10, size=n_days)
    death = np.random.poisson(lam=110 - 0.5 * temp + 0.05 * pm10)
    cvd = np.random.poisson(lam=death * 0.4)
    o3 = np.random.gamma(shape=2, scale=9, size=n_days)
    dptp = temp - np.random.uniform(2, 8, n_days)
    dow = dates.day_name().str[:3]

    df = pd.DataFrame({
        "date": dates, "death": death, "cvd": cvd, "temp": temp,
        "pm10": pm10, "o3": o3, "dptp": dptp, "dow": dow
    })

# 缺失值与描述性统计 (对应 R: miss_var_summary & summary_table)
summary_metrics = []
for var in ['death', 'temp', 'pm10']:
    summary_metrics.append({
        'variable': var,
        'mean': df[var].mean(),
        'p25': df[var].quantile(0.25),
        'p50': df[var].median(),
        'p75': df[var].quantile(0.75),
        'na_n': df[var].isna().sum()
    })
summary_table = pd.DataFrame(summary_metrics)
print("--- 描述性汇总表 ---")
print(summary_table.to_string(index=False))

# ==============================================================================
# 补齐任务 2: 绘制死亡数、温度、PM10 的时间序列图（带平滑长期趋势线）
# 对应 R 语言: ggplot + geom_line + geom_smooth(method="gam") + facet_wrap
# ==============================================================================

# 使用 statsmodels 的 LOWESS（局部加权回归平滑）模拟 R 中的 GAM 平滑线
from statsmodels.nonparametric.smoothers_lowess import lowess

variables = [
    ("death", "Daily Deaths", "firebrick"),
    ("temp", "Temperature (°C)", "darkorange"),
    ("pm10", "PM10 (µg/m³)", "steelblue")
]

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(14, 9), sharex=True)

for ax, (var, label, trend_color) in zip(axes, variables):
    # 1. 过滤缺失值以便计算平滑曲线
    valid_mask = df[var].notna()
    x_dates = df.loc[valid_mask, 'date']
    y_vals = df.loc[valid_mask, var]
    
    # 2. 绘制原始每日波动序列（细折线，半透明）
    ax.plot(x_dates, y_vals, color="grey", alpha=0.5, linewidth=0.5, label="Daily Observations")
    
    # 3. 拟合长期趋势与季节性平滑曲线 (frac 调小保留季节波动，类似 R 中 s(x, bs="cs"))
    # frac=0.05 对应大约几百天的时间平滑窗口，既展示年际季节周期，又展示长期走向
    smooth = lowess(y_vals, np.arange(len(y_vals)), frac=0.03)
    ax.plot(x_dates, smooth[:, 1], color=trend_color, linewidth=1.8, label="Smoothed Trend (LOWESS)")
    
    # 4. 坐标轴与标签美化
    ax.set_ylabel(label, fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=True, fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)

axes[-1].set_xlabel("Date", fontsize=12, fontweight="bold")
plt.suptitle("Time Series of Daily Deaths, Temperature, and PM10 in Chicago (1987–2000)", fontsize=14, fontweight="bold", y=0.99)
plt.tight_layout()
plt.show()

# ==============================================================================
# 模块 3: DLNM 交叉基建模与图表绘制 (对应任务 3)
# ==============================================================================
print("\n>>> [Step 3] 构建温度交叉基并拟合准泊松回归...")

# 构建时间索引变量 (对应 R: df$time <- 1:nrow(df))
df['time'] = np.arange(1, len(df) + 1)

# 构建 DLNM 交叉基 (21天滞后)
cb_engine = DLNMCrossBasis(max_lag=21)
cb_matrix = cb_engine.fit_transform(df['temp'].values)

cb_cols = [f"cb_{i}" for i in range(cb_matrix.shape[1])]
for idx, col in enumerate(cb_cols):
    df[col] = cb_matrix[:, idx]

# 构造长周期时间样条 (7 df/year * 14 years = 98 df)
# 对应 R: ns(time, df = 7 * 14)
time_basis = dmatrix("cr(time, df=98) - 1", {"time": df['time']}, return_type='dataframe')
time_cols = [f"time_spline_{i}" for i in range(time_basis.shape[1])]
time_basis.columns = time_cols
df = pd.concat([df, time_basis], axis=1)

# 拟合准泊松模型 (Quasi-Poisson GLM)
# 对应 R: glm(death ~ cb + ns(time, 98) + dow, family = quasipoisson())
formula = f"death ~ {' + '.join(cb_cols)} + {' + '.join(time_cols)} + C(dow)"
# 使用 statsmodels 的 GLM (Poisson 族，通过非标定 Pearson 卡方捕获过度离散度)
model_dlnm = smf.glm(formula=formula, data=df, family=sm.families.Poisson()).fit(scale='X2')

# 寻找最低死亡温度 (MMT, Minimum Mortality Temperature)
temp_eval = np.arange(df['temp'].min(), df['temp'].max(), 0.5)
pred_initial = cb_engine.predict(model_dlnm, cb_cols, temp_eval)
mmt = temp_eval[np.argmin(pred_initial['allRRfit'])]
print(f"计算得出的最低死亡温度 (MMT): {mmt:.2f} °C")

# 以 MMT 为参照重新进行中心化预测
p1 = df['temp'].quantile(0.01)
p99 = df['temp'].quantile(0.99)
eval_grid = np.unique(np.sort(np.concatenate([temp_eval, [p1, p99, mmt]])))
pred = cb_engine.predict(model_dlnm, cb_cols, eval_grid, cen_val=mmt)

# ----------------- 绘图展示 (对应三张核心图) -----------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图 1: 总体累积暴露-反应曲线 (Overall Cumulative Curve)
axes[0].plot(pred['predvar'], pred['allRRfit'], color='darkred', lw=2, label="Cumulative RR")
axes[0].fill_between(pred['predvar'], pred['allRRlow'], pred['allRRhigh'], color='grey', alpha=0.3, label="95% CI")
axes[0].axhline(1.0, color='black', linestyle='--', lw=0.8)
axes[0].axvline(mmt, color='forestgreen', linestyle=':', label=f"MMT ({mmt:.1f}°C)")
axes[0].set_xlabel("Temperature (°C)")
axes[0].set_ylabel("Cumulative Relative Risk (RR)")
axes[0].set_title(f"Overall Cumulative Association (Ref: {mmt:.1f}°C)")
axes[0].legend()

# 图 2: 极冷 (P1) 与极热 (P99) 滞后反应切片曲线 (Lag-Response Slices)
idx_p1 = np.argmin(np.abs(pred['predvar'] - p1))
idx_p99 = np.argmin(np.abs(pred['predvar'] - p99))

axes[1].plot(pred['lags'], pred['matRRfit'][idx_p1, :], color='blue', lw=2, label=f"Cold P1 ({p1:.1f}°C)")
axes[1].plot(pred['lags'], pred['matRRfit'][idx_p99, :], color='red', lw=2, label=f"Heat P99 ({p99:.1f}°C)")
axes[1].axhline(1.0, color='black', linestyle='--', lw=0.8)
axes[1].set_xlabel("Lag (Days)")
axes[1].set_ylabel("Relative Risk (RR)")
axes[1].set_title(f"Lag-Response Slices (Cold vs. Heat)")
axes[1].legend()
plt.tight_layout()
plt.show()

# 图 3: 2D 等高线曲面图 (Contour Plot)
plt.figure(figsize=(8, 6))
X_mesh, Y_mesh = np.meshgrid(pred['predvar'], pred['lags'])
Z_mesh = pred['matRRfit'].T
cp = plt.contourf(X_mesh, Y_mesh, Z_mesh, levels=20, cmap="coolwarm", extend="both")
plt.colorbar(cp, label="Relative Risk (RR)")
plt.axvline(mmt, color='white', linestyle='--', alpha=0.8, label=f"MMT ({mmt:.1f}°C)")
plt.xlabel("Temperature (°C)")
plt.ylabel("Lag (Days)")
plt.title("Exposure-Lag-Response Contour Surface")
plt.legend()
plt.show()

# 打印极端温度风险
p1_rr = pred['allRRfit'][idx_p1]
p1_low, p1_high = pred['allRRlow'][idx_p1], pred['allRRhigh'][idx_p1]
p99_rr = pred['allRRfit'][idx_p99]
p99_low, p99_high = pred['allRRlow'][idx_p99], pred['allRRhigh'][idx_p99]
print(f"极寒 (P1={p1:.1f}°C): RR = {p1_rr:.4f} (95% CI: {p1_low:.4f}, {p1_high:.4f})")
print(f"极热 (P99={p99:.1f}°C): RR = {p99_rr:.4f} (95% CI: {p99_low:.4f}, {p99_high:.4f})")


# ==============================================================================
# 模块 4: 多变量空气污染回归与敏感性分析 (对应任务 4)
# ==============================================================================
print("\n>>> [Step 4] 运行多变量时间序列模型与敏感性分析...")

# 计算滞后 0-1 天移动均值
df['pm10_lag01'] = (df['pm10'] + df['pm10'].shift(1)) / 2
df['o3_lag01'] = (df['o3'] + df['o3'].shift(1)) / 2
df['season'] = np.where(df['date'].dt.month.isin(range(5, 11)), 'Warm', 'Cold')

# 辅助函数: 计算暴露每升高 10 单位的 RR 及 95% CI
def extract_rr_10(fit_model, var_name, model_label="Model"):
    beta = fit_model.params[var_name]
    se = fit_model.bse[var_name]
    rr = np.exp(beta * 10)
    rr_low = np.exp((beta - 1.96 * se) * 10)
    rr_high = np.exp((beta + 1.96 * se) * 10)
    return {
        "Model": model_label,
        "RR": round(rr, 4),
        "95% CI": f"{rr:.4f} ({rr_low:.4f}, {rr_high:.4f})"
    }

# 4.1 主模型: PM10 (lag 0-1) + 温度 DLNM 交叉基
m4_base = smf.glm(f"death ~ pm10_lag01 + {' + '.join(cb_cols)} + {' + '.join(time_cols)} + C(dow)",
                  data=df, family=sm.families.Poisson()).fit(scale='X2')

# 4.2 双污染物模型: 加入 O3
m4_dual = smf.glm(f"death ~ pm10_lag01 + o3_lag01 + {' + '.join(cb_cols)} + {' + '.join(time_cols)} + C(dow)",
                  data=df, family=sm.families.Poisson()).fit(scale='X2')

# 4.3 敏感性分析: 调整时间样条自由度 (每年 5 df vs 每年 10 df)
time_df5 = dmatrix("cr(time, df=70) - 1", {"time": df['time']}, return_type='dataframe')
time_df5.columns = [f"t_df5_{i}" for i in range(time_df5.shape[1])]
df_sens = pd.concat([df, time_df5], axis=1)

m4_df5 = smf.glm(f"death ~ pm10_lag01 + {' + '.join(cb_cols)} + {' + '.join(time_df5.columns)} + C(dow)",
                 data=df_sens, family=sm.families.Poisson()).fit(scale='X2')

results_multivar = pd.DataFrame([
    extract_rr_10(m4_base, "pm10_lag01", "1. Single-Pollutant (PM10)"),
    extract_rr_10(m4_dual, "pm10_lag01", "2. Dual-Pollutant (+ O3)"),
    extract_rr_10(m4_df5, "pm10_lag01", "3. Sensitivity (df=5/year)")
])
print(results_multivar.to_string(index=False))


# ==============================================================================
# 模块 5: 时间分层病例交叉设计 (Case-Crossover via Conditional Poisson)
# ==============================================================================
print("\n>>> [Step 5] 运行时间分层病例交叉设计 (Conditional Poisson)...")

# 【教学重点】：
# R 中 gnm::gnm(death ~ pm10 + cb, eliminate = stratum) 将层效应 (stratum) 吸收。
# 在统计数学上，条件泊松似然等价于对每个 stratum (年-月-星期几) 内部消除截距。
# 在 Python 中，可通过 linearmodels 的 Panel 泊松回归，或利用带固定效应的广义模型求解。
# 此处我们构建每个层内的期望调整，或者使用条件分组消除算法。

df['stratum'] = (df['date'].dt.year.astype(str) + ":" +
                 df['date'].dt.month.astype(str) + ":" +
                 df['dow'].astype(str))

# 为保持纯轻量级标准库依赖，此处使用 Statsmodels 的广义泊松回归，
# 并通过群组聚类或高维哑变量/吸收因子拟合（对应 R 的 eliminate 功能）
# 针对心血管死亡 (cvd) 与全死因 (death) 分别评估
m5_cc_death = smf.glm(f"death ~ pm10_lag01 + {' + '.join(cb_cols)} + C(stratum)",
                      data=df.dropna(subset=['pm10_lag01', 'cb_0']), 
                      family=sm.families.Poisson()).fit(scale='X2')

m5_cc_cvd = smf.glm(f"cvd ~ pm10_lag01 + {' + '.join(cb_cols)} + C(stratum)",
                    data=df.dropna(subset=['pm10_lag01', 'cb_0']), 
                    family=sm.families.Poisson()).fit(scale='X2')

# 时间序列对应的 cvd 模型对照
m5_ts_cvd = smf.glm(f"cvd ~ pm10_lag01 + {' + '.join(cb_cols)} + {' + '.join(time_cols)} + C(dow)",
                    data=df, family=sm.families.Poisson()).fit(scale='X2')

# 最终方法学对比表
comparison_table = pd.DataFrame([
    {"Design": "Time-Series", "Outcome": "All-cause Death", 
     "RR (95% CI)": extract_rr_10(m4_base, "pm10_lag01")["95% CI"]},
    {"Design": "Case-Crossover", "Outcome": "All-cause Death", 
     "RR (95% CI)": extract_rr_10(m5_cc_death, "pm10_lag01")["95% CI"]},
    {"Design": "Time-Series", "Outcome": "Cardiovascular (CVD)", 
     "RR (95% CI)": extract_rr_10(m5_ts_cvd, "pm10_lag01")["95% CI"]},
    {"Design": "Case-Crossover", "Outcome": "Cardiovascular (CVD)", 
     "RR (95% CI)": extract_rr_10(m5_cc_cvd, "pm10_lag01")["95% CI"]}
])

print("\n--- 时间序列 vs 病例交叉设计 对比结果 ---")
print(comparison_table.to_string(index=False))