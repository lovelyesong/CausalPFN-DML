import os, numpy as np, pandas as pd
from doubleml.datasets import fetch_bonus

def main():
    out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'bonus'))
    os.makedirs(out_dir, exist_ok=True)

    dml_data = fetch_bonus()                 # DoubleMLData
    df = dml_data.data.copy()
    x_cols = dml_data.x_cols
    t_col  = dml_data.d_cols[0]
    y_col  = dml_data.y_col

    X = df[x_cols].to_numpy(dtype=np.float32)
    T = df[t_col].to_numpy(dtype=np.float32)
    Y = df[y_col].to_numpy(dtype=np.float32)

    df.to_csv(os.path.join(out_dir, 'bonus_raw.csv'), index=False)
    np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'T.npy'), T)
    np.save(os.path.join(out_dir, 'Y.npy'), Y)

    meta = {'x_cols': x_cols, 't_col': t_col, 'y_col': y_col, 'n': int(X.shape[0]), 'p': int(X.shape[1])}
    pd.Series(meta, dtype=object).to_json(os.path.join(out_dir, 'meta.json'), force_ascii=False)

    print(f"[OK] saved in {out_dir} | n={X.shape[0]}, p={X.shape[1]}")
if __name__ == "__main__":
    main()
