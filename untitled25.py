# -*- coding: utf-8 -*-
"""ロジスティック回帰（二項GLM）＆多重比較アプリ（改良版）"""
import itertools
import warnings
from io import StringIO, BytesIO

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multitest import multipletests

# ---- 日本語フォント（japanize_matplotlib が無くても落ちない）----
try:
    import japanize_matplotlib  # noqa: F401
except Exception:
    pass
sns.set_theme(style="whitegrid")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = [
    "IPAexGothic", "IPAGothic", "Noto Sans CJK JP", "Yu Gothic", "Meiryo",
    "Hiragino Sans", "TakaoPGothic",
] + list(plt.rcParams["font.sans-serif"])
plt.rcParams["axes.unicode_minus"] = False

ALPHA = 0.05
FAC = "カテゴリ (Factor)"
NUM = "数量 (Numeric)"
MC_GLM = "GLM上のWald検定 + Holm補正（推奨）"
MC_TUKEY = "Tukey HSD（割合に対する従来法・非推奨）"


# =========================================================
# ユーティリティ
# =========================================================
def q(name):
    """patsy用に列名を安全に引用（空白・日本語・記号入りの列名対策）"""
    return 'Q("' + str(name).replace("\\", "\\\\").replace('"', '\\"') + '")'


def natural_levels(s):
    """水準を数値として並べられるなら数値順、そうでなければ文字列順（'10' < '2' 問題の回避）"""
    vals = pd.unique(s.astype(str))
    try:
        return sorted(vals, key=float)
    except ValueError:
        return sorted(vals)


def load_uploaded(f):
    raw = f.getvalue()
    for enc in ("utf-8-sig", "cp932"):
        try:
            return pd.read_csv(StringIO(raw.decode(enc)))
        except UnicodeDecodeError:
            continue
    raise ValueError("文字コードを判別できません（UTF-8 / Shift_JIS に対応）")


def load_pasted(text):
    df = pd.read_csv(StringIO(text), sep="\t")
    if df.shape[1] == 1:  # タブでなくカンマ区切りだった場合
        df = pd.read_csv(StringIO(text), sep=",")
    return df


def fig_to_image(fig, width_px):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)  # メモリリーク防止
    st.image(buf.getvalue(), width=width_px)


def fit_glm(d, terms, quasi):
    rhs = " + ".join(terms) if terms else "1"
    return smf.glm(
        f"_prop ~ {rhs}", data=d,
        family=sm.families.Binomial(),
        var_weights=d["_n"].to_numpy(),
    ).fit(scale="X2" if quasi else None)


def make_cld(levels, means, pw):
    """有意差なしの関係から maximal clique を求め、アルファベットを割り当てる"""
    G = nx.Graph()
    G.add_nodes_from(levels)
    for _, r in pw.iterrows():
        if not r["reject"]:
            G.add_edge(r["group1"], r["group2"])
    cliques = sorted(
        nx.find_cliques(G),
        key=lambda c: (-np.mean([means[g] for g in c]), sorted(c)),
    )
    cld = {g: "" for g in levels}
    for j, c in enumerate(cliques):
        for g in c:
            cld[g] += chr(ord("a") + j % 26)
    return {g: "".join(sorted(v)) for g, v in cld.items()}


def glm_pairwise(model, term, levels, method="holm"):
    """GLMの係数（logit尺度）の差をWald検定し、多重比較補正する"""
    names = list(model.params.index)
    beta = model.params.to_numpy()
    cov = model.cov_params().to_numpy()

    def vec(lv):
        v = np.zeros(len(names))
        nm = f"{term}[T.{lv}]"
        if nm in names:  # 基準水準は0のまま
            v[names.index(nm)] = 1.0
        return v

    rows = []
    for a, b in itertools.combinations(levels, 2):
        c = vec(b) - vec(a)
        est = float(c @ beta)
        se = float(np.sqrt(c @ cov @ c))
        z = est / se if se > 0 else np.nan
        p = 2 * stats.norm.sf(abs(z)) if np.isfinite(z) else 1.0
        rows.append((a, b, est, se, z, p))
    pw = pd.DataFrame(rows, columns=["group1", "group2", "est", "se", "z", "p_raw"])
    pw["p_adj"] = multipletests(pw["p_raw"], method=method)[1]
    pw["reject"] = pw["p_adj"] < ALPHA
    disp = pd.DataFrame({
        "比較 (後 − 前)": pw["group2"] + " − " + pw["group1"],
        "差 (logit)": pw["est"], "SE": pw["se"], "z": pw["z"],
        "p値": pw["p_raw"], "補正p値 (Holm)": pw["p_adj"],
        "オッズ比": np.exp(pw["est"]), "有意": pw["reject"],
    })
    return pw, disp


def tukey_pairwise(d, var):
    res = pairwise_tukeyhsd(endog=d["_prop"], groups=d[var].astype(str), alpha=ALPHA)
    t = res._results_table.data
    raw = pd.DataFrame(t[1:], columns=t[0]).rename(columns={"p-adj": "p_adj"})
    raw["reject"] = raw["reject"].astype(bool)
    raw["group1"] = raw["group1"].astype(str)
    raw["group2"] = raw["group2"].astype(str)
    disp = raw.rename(columns={"meandiff": "割合の差", "p_adj": "補正p値",
                               "lower": "下限", "upper": "上限", "reject": "有意"})
    return raw, disp


# =========================================================
# 解析本体
# =========================================================
def run_analysis(df, success_col, total_col, var_types, quasi, mc_method):
    if success_col == total_col:
        raise ValueError("成功数と全体数に同じ列が選ばれています。")
    if set(var_types) & {success_col, total_col}:
        raise ValueError("説明変数に成功数・全体数の列を含めることはできません。")

    d = pd.DataFrame({
        "_succ": pd.to_numeric(df[success_col], errors="coerce"),
        "_n": pd.to_numeric(df[total_col], errors="coerce"),
    })
    for v, t in var_types.items():
        d[v] = pd.to_numeric(df[v], errors="coerce") if t == NUM else df[v]
    n_in = len(d)
    d = d.dropna().reset_index(drop=True)
    n_drop = n_in - len(d)
    if d.empty:
        raise ValueError("有効な行がありません（欠損・非数値を確認してください）。")
    if (d["_n"] <= 0).any():
        raise ValueError("全体数に 0 以下の値があります。")
    if ((d["_succ"] < 0) | (d["_succ"] > d["_n"])).any():
        raise ValueError("成功数が 0 未満、または全体数を超えている行があります。")
    d["_prop"] = d["_succ"] / d["_n"]

    levels, terms = {}, {}
    for v, t in var_types.items():
        if t == FAC:
            lv = natural_levels(d[v])
            if len(lv) < 2:
                raise ValueError(f"[{v}] は水準が1つしかありません。")
            d[v] = pd.Categorical(d[v].astype(str), categories=lv)
            levels[v] = lv
            terms[v] = f"C({q(v)})"
        else:
            if d[v].nunique() < 2:
                raise ValueError(f"[{v}] は値が1種類しかありません。")
            terms[v] = q(v)

    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        model = fit_glm(d, list(terms.values()), quasi)
    msgs = sorted({str(w.message) for w in wlist})

    notes = []
    if n_drop:
        notes.append(f"欠損・非数値のため {n_drop} 行を除外しました。")
    if not model.converged:
        notes.append("GLMが収束していません（完全分離の可能性）。")
    if np.linalg.matrix_rank(model.model.exog) < model.model.exog.shape[1]:
        notes.append("説明変数が共線形（ランク落ち）です。結果は信頼できません。")
    notes += [f"statsmodels警告: {m}" for m in msgs]

    # 過分散（Pearson χ² / 残差df）
    phi = float(model.pearson_chi2 / model.df_resid) if model.df_resid > 0 else np.nan

    # 係数表（オッズ比つき）
    mapping = {"Intercept": "Intercept"}
    for v, t in var_types.items():
        if t == FAC:
            for lv in levels[v][1:]:
                mapping[f"{terms[v]}[T.{lv}]"] = f"{v}: {lv}（基準 {levels[v][0]}）"
        else:
            mapping[terms[v]] = v
    ci = model.conf_int()
    coef = pd.DataFrame({
        "係数 (logit)": model.params, "SE": model.bse, "z": model.tvalues,
        "p値": model.pvalues, "オッズ比": np.exp(model.params),
        "OR下限": np.exp(ci[0]), "OR上限": np.exp(ci[1]),
    })
    coef.index = [mapping.get(i, i) for i in coef.index]
    coef.index.name = "項"

    # 変数ごとの全体検定（尤度比 / quasiならF検定）
    lr_rows = []
    for v in var_types:
        red = fit_glm(d, [t for k, t in terms.items() if k != v], quasi)
        ddf = red.df_resid - model.df_resid
        dev = red.deviance - model.deviance
        if quasi:
            f = (dev / ddf) / model.scale
            p = stats.f.sf(f, ddf, model.df_resid)
            lr_rows.append((v, ddf, f, p, "F"))
        else:
            p = stats.chi2.sf(dev, ddf)
            lr_rows.append((v, ddf, dev, p, "χ²"))
    lr = pd.DataFrame(lr_rows, columns=["変数", "自由度", "統計量", "p値", "検定"])

    # 多重比較（カテゴリ変数ごと）
    pairwise = {}
    for v in levels:
        g = d.groupby(v, observed=True)[["_succ", "_n"]].sum()
        means = (g["_succ"] / g["_n"])
        means.index = means.index.astype(str)
        if mc_method == MC_GLM:
            raw, disp = glm_pairwise(model, terms[v], levels[v])
        else:
            raw, disp = tukey_pairwise(d, v)
        cld = make_cld(levels[v], means, raw)
        pairwise[v] = {"raw": raw, "disp": disp, "cld": cld, "means": means}

    return {
        "d": d, "model": model, "formula": model.model.formula,
        "var_types": var_types, "levels": levels, "terms": terms,
        "coef": coef, "lr": lr, "phi": phi, "quasi": quasi,
        "notes": notes, "pairwise": pairwise, "mc_method": mc_method,
    }


# =========================================================
# 描画
# =========================================================
def plot_factor(ax, res, var):
    d, lv, cld = res["d"], res["levels"][var], res["pairwise"][var]["cld"]
    sns.boxplot(data=d, x=var, y="_prop", order=lv, color="#9fd3c0",
                fliersize=0, ax=ax)
    sns.stripplot(data=d, x=var, y="_prop", order=lv, color="#333333",
                  size=3, alpha=0.7, ax=ax)
    top = min(float(d["_prop"].max()), 1.0) + 0.03
    for i, g in enumerate(lv):
        ax.text(i, top, cld[g], ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_ylim(-0.05, 1.15)
    ax.set_title(f"箱ひげ図: {var}", fontsize=10)
    if len(lv) > 5:
        ax.tick_params(axis="x", rotation=45)


def plot_numeric(ax, res, var):
    d, model = res["d"], res["model"]
    vt, levels = res["var_types"], res["levels"]
    ax.scatter(d[var], d["_prop"], s=20 + 80 * d["_n"] / d["_n"].max(),
               alpha=0.7, edgecolor="k", linewidth=0.3)
    x = np.linspace(d[var].min(), d[var].max(), 200)
    grid = pd.DataFrame({var: x})
    for v, t in vt.items():  # 他の変数は 平均 / 最頻水準 に固定
        if v == var:
            continue
        if t == NUM:
            grid[v] = d[v].mean()
        else:
            grid[v] = pd.Categorical([d[v].mode().iloc[0]] * len(x), categories=levels[v])
    sf = model.get_prediction(grid).summary_frame()
    ax.plot(x, sf["mean"], color="red", linewidth=2)
    ax.fill_between(x, sf["mean_ci_lower"], sf["mean_ci_upper"], color="red", alpha=0.15)
    ax.set_ylim(-0.05, 1.05)
    ttl = f"ロジスティック曲線: {var}"
    if len(vt) > 1:
        ttl += "\n(他変数は平均/最頻水準で固定)"
    ax.set_title(ttl, fontsize=9)


def render(res, gw, gh):
    model = res["model"]

    st.header("📊 1. 一般化線形モデル (二項GLM)")
    st.code(f"Model: {res['formula']}\nfamily=Binomial, var_weights=全体数"
            + ("\n※ quasi-binomial（Pearson χ²でdispersion推定）" if res["quasi"] else ""),
            language="text")
    for n in res["notes"]:
        st.warning(n)
    phi = res["phi"]
    if np.isfinite(phi):
        st.metric("過分散指標 (Pearson χ² / 残差df)", f"{phi:.2f}")
        if phi > 1.5 and not res["quasi"]:
            st.warning("過分散の疑い（目安 >1.5）。SEが過小評価され、p値が甘くなっている可能性があります。"
                       "サイドバーの「quasi-binomial」をONにするか、ブロック/反復を変量効果に入れるモデルを検討してください。")
    else:
        st.info("残差dfが0（飽和モデル）のため過分散は評価できません。")

    st.subheader("係数表")
    st.dataframe(res["coef"])
    st.download_button("係数表をCSVで保存",
                       res["coef"].to_csv().encode("utf-8-sig"), "glm_coef.csv", "text/csv")
    st.subheader("変数ごとの全体検定")
    st.dataframe(res["lr"], hide_index=True)
    with st.expander("statsmodels summary（原文）"):
        st.text(model.summary())
    st.info("""
**【読み方】**
- **係数 (logit)**: 基準水準（または数量変数が1増えたとき）に対する対数オッズの差。正なら割合を上げ、負なら下げます。
- **オッズ比**: exp(係数)。1より大きければ割合が上がる方向。信頼区間が1をまたげば有意ではありません。
- **Intercept**: 全カテゴリ変数が基準水準、数量変数が0のときの対数オッズ（数量変数が0を含まない範囲なら外挿値）。
- **全体検定**: 3水準以上のカテゴリ変数は、個別の係数p値ではなくこちらで「変数として効いているか」を判断します。
""")

    st.header("📈 2. 変数ごとのグラフ")
    vars_ = list(res["var_types"])
    px = int(gw * 110)
    for start in range(0, len(vars_), 3):
        cols = st.columns(3)
        for col, var in zip(cols, vars_[start:start + 3]):
            with col:
                fig, ax = plt.subplots(figsize=(gw, gh))
                if res["var_types"][var] == FAC:
                    plot_factor(ax, res, var)
                else:
                    plot_numeric(ax, res, var)
                ax.set_ylabel("割合 (Proportion)", fontsize=9)
                ax.set_xlabel(var, fontsize=9)
                ax.tick_params(axis="both", labelsize=8)
                fig_to_image(fig, px)

    st.header("🔠 3. 多重比較 (カテゴリ変数のみ)")
    if not res["pairwise"]:
        st.info("カテゴリ(Factor)の説明変数がないため、多重比較はスキップしました。")
        return
    if res["mc_method"] == MC_TUKEY:
        st.warning("Tukey HSDは割合を正規分布・等分散として扱い、全体数 n の違いも他の説明変数も無視します。"
                   "上のGLMと整合しないため、参考値としてのみ使ってください。")
    for var, pr in res["pairwise"].items():
        st.subheader(f"［{var}］")
        st.dataframe(pr["disp"], hide_index=True)
        cld_df = pd.DataFrame({
            "Group": pr["means"].index,
            "Pooled proportion": pr["means"].values,
            "Significance (CLD)": [pr["cld"][g] for g in pr["means"].index],
        }).sort_values("Pooled proportion", ascending=False)
        st.markdown("**Compact Letter Display**（同じ文字を共有する群間は有意差なし）")
        st.dataframe(cld_df, hide_index=True)
    st.caption("GLM法: 他の説明変数を調整したlogit尺度の差をWald検定し、Holm法で補正。"
               "CLD表の割合は群ごとの生データのプール値（Σ成功数/Σ全体数）です。")


# =========================================================
# メイン
# =========================================================
def main():
    st.set_page_config(page_title="ロジスティック回帰＆多重比較アプリ", layout="wide")
    st.title("🌱 ロジスティック回帰＆多重比較アプリ")

    st.sidebar.header("1. データの入力")
    method = st.sidebar.radio("データ入力方法を選択", ["Excelからコピペ", "CSVアップロード"])
    df = None
    try:
        if method == "CSVアップロード":
            f = st.sidebar.file_uploader("CSVファイルを選択", type=["csv"])
            if f is not None:
                df = load_uploaded(f)
        else:
            st.sidebar.markdown("Excelの表をコピーして貼り付けてください")
            txt = st.sidebar.text_area("データ貼り付けエリア", height=150)
            if txt.strip():
                df = load_pasted(txt.strip())
    except Exception as e:
        st.error(f"データの読み込みに失敗しました: {e}")

    if df is None or df.empty:
        st.info("左側のサイドバーからデータを入力してください。")
        return

    df.columns = [str(c).strip() for c in df.columns]
    if df.columns.duplicated().any():
        st.error("列名が重複しています。列名を一意にしてください。")
        return

    st.subheader("データプレビュー")
    st.dataframe(df.head())

    st.sidebar.header("2. 変数の設定")
    cols = df.columns.tolist()
    idx = lambda name, default: cols.index(name) if name in cols else default
    success_col = st.sidebar.selectbox("成功数（例: sprouted）", cols, index=idx("sprouted", 0))
    total_col = st.sidebar.selectbox("全体数（例: n）", cols,
                                     index=idx("n", 1 if len(cols) > 1 else 0))
    cand = [c for c in cols if c not in (success_col, total_col)]
    selected = st.sidebar.multiselect(
        "説明変数を選択", cand, default=[v for v in ("var", "water") if v in cand])

    var_types = {}
    if selected:
        st.sidebar.markdown("**説明変数の扱い:**")
        for v in selected:
            is_num = pd.api.types.is_numeric_dtype(df[v]) and df[v].nunique() >= 6
            var_types[v] = st.sidebar.radio(
                f"[{v}] のデータ型", [FAC, NUM], index=1 if is_num else 0, key=f"type_{v}")

    st.sidebar.markdown("---")
    st.sidebar.header("3. 解析オプション")
    quasi = st.sidebar.checkbox("quasi-binomial（過分散を補正）", value=False)
    mc_method = st.sidebar.radio("多重比較の方法", [MC_GLM, MC_TUKEY])

    st.sidebar.markdown("---")
    st.sidebar.header("4. グラフの表示設定")
    gw = st.sidebar.slider("グラフの横幅", 2.0, 8.0, 3.5, 0.5)
    gh = st.sidebar.slider("グラフの高さ", 2.0, 8.0, 3.0, 0.5)

    sig = (tuple(cols), df.shape, method)
    if st.sidebar.button("解析を実行する"):
        if not selected:
            st.warning("説明変数を1つ以上選択してください。")
        else:
            try:
                with st.spinner("解析中..."):
                    st.session_state["result"] = run_analysis(
                        df, success_col, total_col, var_types, quasi, mc_method)
                    st.session_state["sig"] = sig
            except ValueError as e:
                st.session_state.pop("result", None)
                st.error(str(e))
            except Exception as e:
                st.session_state.pop("result", None)
                st.error(f"解析中にエラーが発生しました: {type(e).__name__}: {e}")

    # 結果はsession_stateに保持 → スライダー操作で結果が消えない
    res = st.session_state.get("result")
    if res is not None and st.session_state.get("sig") == sig:
        st.caption("※ 変数・オプションを変更した場合は「解析を実行する」を再度押してください。")
        render(res, gw, gh)


if __name__ == "__main__":
    main()