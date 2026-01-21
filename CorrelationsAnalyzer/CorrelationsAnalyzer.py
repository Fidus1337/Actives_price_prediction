import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


class CorrelationsAnalyzer:
    """Класс для анализа корреляций фичей с целевой переменной."""

    def bh_fdr(self, pvals: pd.Series) -> pd.Series:
        """Benjamini–Hochberg FDR."""
        p = pvals.dropna()
        n = len(p)
        if n == 0:
            return pvals
        order = np.argsort(p.values)
        ranked = p.values[order]
        q = ranked * n / (np.arange(1, n + 1))
        q = np.minimum.accumulate(q[::-1])[::-1]
        out = pd.Series(np.nan, index=pvals.index, dtype="float64")
        out.loc[p.index[order]] = np.clip(q, 0, 1)
        return out

    def corr_report(self, df: pd.DataFrame, method: str = "pearson", min_n: int = 60) -> pd.DataFrame:
        """
        Расчёт корреляции всех числовых фичей с y_up_1d.
        Возвращает DataFrame с колонками: feature, corr, abs_corr, n, p_value, q_value_fdr.
        """
        tmp = df.copy()
        tmp["y_up_1d"] = pd.to_numeric(tmp["y_up_1d"], errors="coerce")

        features = [
            c for c in tmp.columns
            if c not in {"date", "y_up_1d"}
            and pd.api.types.is_numeric_dtype(tmp[c])
        ]

        # быстрый corr (без p-values)
        corr = tmp[features].corrwith(tmp["y_up_1d"], method=method)
        res = corr.rename("corr").to_frame()
        res["abs_corr"] = res["corr"].abs()

        # n по каждой фиче
        y = tmp["y_up_1d"]
        n_list = []
        for c in features:
            m = tmp[c].notna() & y.notna()
            n_list.append(int(m.sum()))
        res["n"] = n_list

        # p-values
        pvals = []
        for c in features:
            m = tmp[c].notna() & y.notna()
            if m.sum() < min_n:
                pvals.append(np.nan)
                continue
            x = pd.to_numeric(tmp.loc[m, c], errors="coerce")
            yy = tmp.loc[m, "y_up_1d"]
            if method == "pearson":
                _, p = pearsonr(x, yy)
            else:
                _, p = spearmanr(x, yy)
            pvals.append(p)
        res["p_value"] = pvals
        res["q_value_fdr"] = self.bh_fdr(res["p_value"])

        res = (
            res.reset_index()
               .rename(columns={"index": "feature"})
               .sort_values("abs_corr", ascending=False)
               .reset_index(drop=True)
        )
        return res

    def corr_table_with_pvalues(self, df: pd.DataFrame, target: str = "y_up_1d", method: str = "pearson") -> pd.DataFrame:
        """
        Расчёт корреляции с p-values для всех числовых фичей.
        Возвращает DataFrame с колонками: feature, corr, p_value, n, abs_corr, q_value_fdr.
        """
        y = pd.to_numeric(df[target], errors="coerce")
        num_cols = [c for c in df.columns if c not in {"date", target} and pd.api.types.is_numeric_dtype(df[c])]
        rows = []
        for c in num_cols:
            x = pd.to_numeric(df[c], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() < 30:
                continue
            if method == "pearson":
                r, p = pearsonr(x[m], y[m])
            else:
                r, p = spearmanr(x[m], y[m])
            rows.append((c, r, p, m.sum()))
        res = pd.DataFrame(rows, columns=["feature", "corr", "p_value", "n"])
        res["abs_corr"] = res["corr"].abs()
        res["q_value_fdr"] = self.bh_fdr(res["p_value"])
        return res.sort_values("abs_corr", ascending=False).reset_index(drop=True)

    def group_effect_report(self, df: pd.DataFrame, target: str = "y_up_1d", top_n: int = 30) -> pd.DataFrame:
        """
        Расчёт эффекта (Cohen's d) для каждой фичи между группами y=0 и y=1.
        Возвращает DataFrame с колонками: feature, mean_y1, mean_y0, lift, abs_cohen_d, cohen_d, n_y1, n_y0.
        """
        y = pd.to_numeric(df[target], errors="coerce")
        feats = [c for c in df.columns if c not in {"date", target} and pd.api.types.is_numeric_dtype(df[c])]

        rows = []
        for f in feats:
            x = pd.to_numeric(df[f], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() < 60:
                continue
            x0 = x[m & (y == 0)]
            x1 = x[m & (y == 1)]
            if len(x0) < 20 or len(x1) < 20:
                continue

            mean0, mean1 = float(x0.mean()), float(x1.mean())
            std0, std1 = float(x0.std(ddof=1)), float(x1.std(ddof=1))
            pooled = np.sqrt(((len(x0) - 1) * std0**2 + (len(x1) - 1) * std1**2) / (len(x0) + len(x1) - 2))
            d = (mean1 - mean0) / (pooled + 1e-12)

            rows.append((f, mean1, mean0, mean1 - mean0, abs(d), d, len(x1), len(x0)))

        res = pd.DataFrame(rows, columns=["feature", "mean_y1", "mean_y0", "lift", "abs_cohen_d", "cohen_d", "n_y1", "n_y0"])
        return res.sort_values("abs_cohen_d", ascending=False).head(top_n).reset_index(drop=True)
