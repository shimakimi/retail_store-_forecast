import streamlit as st
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ページ設定
st.set_page_config(page_title="Sales Forecast Demo", layout="wide")
sns.set(style="whitegrid")


#  データ読み込みと前処理 (キャッシュ化)
@st.cache_data
def load_and_process_data():
    """データ読み込みから特徴量作成までを行う関数"""
    df = pd.read_csv('dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['store', 'item', 'date']).reset_index(drop=True)
    
    # 予測基準日 (Anchor Date: T) の定義
    target_month_days = [(6, 1), (6, 30), (12, 1)]
    years = range(2013, 2018)

    anchor_dates = []
    for y in years:
        for m, d in target_month_days:
            anchor_dates.append(pd.Timestamp(year=y, month=m, day=d))

    # Base DataFrame作成
    unique_store_items = df[['store', 'item']].drop_duplicates()
    base_df = pd.merge(
        unique_store_items.assign(key=1),
        pd.DataFrame({'date_T': anchor_dates, 'key': 1}),
        on='key'
    ).drop('key', axis=1)

    # 特徴量 (X) 作成
    X_features = base_df.copy()
    
    # ラグ特徴量
    lags = [0, 1, 7, 14, 28, 365]
    for lag in lags:
        lag_date_col = f'date_T_minus_{lag}'
        X_features[lag_date_col] = X_features['date_T'] - timedelta(days=lag)
        X_features = pd.merge(
            X_features,
            df[['date', 'store', 'item', 'sales']].rename(columns={'sales': f'lag_{lag}'}),
            left_on=[lag_date_col, 'store', 'item'],
            right_on=['date', 'store', 'item'],
            how='left'
        ).drop(columns=[lag_date_col, 'date'])

    # 移動平均特徴量
    df_rolled = df.copy()
    df_rolled['roll_mean_7'] = df_rolled.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(7).mean())
    df_rolled['roll_mean_28'] = df_rolled.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(28).mean())
    df_rolled['roll_std_7'] = df_rolled.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(7).std())

    X_features = pd.merge(
        X_features,
        df_rolled[['date', 'store', 'item', 'roll_mean_7', 'roll_mean_28', 'roll_std_7']],
        left_on=['date_T', 'store', 'item'],
        right_on=['date', 'store', 'item'],
        how='left'
    ).drop(columns=['date'])

    # 日付特徴量
    X_features['month'] = X_features['date_T'].dt.month
    X_features['year'] = X_features['date_T'].dt.year

    # ターゲット (Y) 作成
    target_dfs = []
    for i in range(7, 29):
        col_name = f'sales_T+{i}'
        tmp = base_df[['date_T', 'store', 'item']].copy()
        tmp['target_date'] = tmp['date_T'] + timedelta(days=i)
        merged = pd.merge(
            tmp, df[['date', 'store', 'item', 'sales']],
            left_on=['target_date', 'store', 'item'],
            right_on=['date', 'store', 'item'], how='left'
        )
        tmp['sales'] = merged['sales']
        tmp['days_ahead'] = col_name
        target_dfs.append(tmp)

    Y_features = pd.concat(target_dfs).pivot_table(
        index=['date_T', 'store', 'item'], columns='days_ahead', values='sales'
    ).reset_index()

    # 結合
    final_df = pd.merge(X_features, Y_features, on=['date_T', 'store', 'item'])
    
    # 欠損値処理（シンプルに0埋め、または削除）
    final_df = final_df.fillna(0) 

    return final_df


# 2モデル学習
@st.cache_resource
def train_models(data):
    """22個のLightGBMモデルを一括学習"""
    train_df = data[data['year'] < 2017]
    
    # 特徴量カラム定義
    exclude_cols = ['date_T', 'store', 'item', 'year', 'month']
    feature_cols = [c for c in train_df.columns if c not in exclude_cols and 'sales_T+' not in c]
    feature_cols += ['month']
    
    target_cols = [f'sales_T+{i}' for i in range(7, 29)]
    
    models = {}
    
    params = {
        'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
        'learning_rate': 0.05, 'num_leaves': 31, 'feature_fraction': 0.8,
        'bagging_fraction': 0.8, 'bagging_freq': 5, 'seed': 42, 'verbosity': -1
    }

    # プログレスバー表示
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, target in enumerate(target_cols):
        days_ahead = target.split('_')[-1]
        status_text.text(f"Training model for {days_ahead}...")
        
        X_train = train_df[train_df['year'] < 2016][feature_cols]
        y_train = train_df[train_df['year'] < 2016][target]
        X_val = train_df[train_df['year'] == 2016][feature_cols]
        y_val = train_df[train_df['year'] == 2016][target]

        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

        model = lgb.train(
            params, lgb_train, num_boost_round=500,
            valid_sets=[lgb_train, lgb_val],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        models[target] = model
        progress_bar.progress((idx + 1) / len(target_cols))

    status_text.text("Training Complete!")
    progress_bar.empty()
    
    return models, feature_cols, target_cols


# メインアプリケーション処理
st.title("ボーナス商戦シュミレーション AI")
st.markdown("機械学習を用いたボーナス商戦（6月・7月・12月）の売上予測シミュレーター")

# 1. データ読み込み
with st.spinner('Loading data...'):
    data = load_and_process_data()

# 2. モデル学習（初回のみ実行）
with st.spinner('Training models... (This may take a moment)'):
    models, feature_cols, target_cols = train_models(data)

# テストデータ (2017年) の準備
test_df = data[data['year'] == 2017].copy()


# UI: サイドバー設定
st.sidebar.header("Filter Settings")
selected_store = st.sidebar.selectbox("Select Store", sorted(test_df['store'].unique()))
selected_item = st.sidebar.selectbox("Select Item", sorted(test_df['item'].unique()))
selected_season = st.sidebar.radio("Target Season", ["June (6/1 Base)", "July (6/30 Base)", "Dec (12/1 Base)"])

# 基準日の特定
season_map = {
    "June (6/1 Base)": "2017-06-01",
    "July (6/30 Base)": "2017-06-30",
    "Dec (12/1 Base)": "2017-12-01"
}
target_date_str = season_map[selected_season]
target_date_ts = pd.Timestamp(target_date_str)

# 対象データの抽出（1行のみ）
current_row = test_df[
    (test_df['date_T'] == target_date_ts) &
    (test_df['store'] == selected_store) &
    (test_df['item'] == selected_item)
]

if current_row.empty:
    st.error("Data not found for selection.")
    st.stop()


# タブ構成
tab1, tab2, tab3 = st.tabs(["📈 Forecast Result", "🎛 Simulation (What-If)", "🔍 Feature Importance"])

# --- Tab 1: 予測結果の可視化 ---
with tab1:
    st.subheader(f"Forecast for Store {selected_store}, Item {selected_item} ({selected_season})")
    
    # 予測実行
    preds = []
    actuals = []
    days_labels = []

    input_data = current_row[feature_cols]

    for target in target_cols:
        pred_val = models[target].predict(input_data)[0]
        actual_val = current_row[target].values[0]
        
        preds.append(pred_val)
        actuals.append(actual_val)
        days_labels.append(target.replace("sales_", ""))

    # データフレーム化
    res_df = pd.DataFrame({
        "Days Ahead": days_labels,
        "Actual Sales": actuals,
        "Predicted Sales": preds
    })
    
    # 日付列を追加（グラフ用）
    res_df['Date'] = [target_date_ts + timedelta(days=int(d.split('+')[1])) for d in days_labels]

    # メトリクス表示
    col1, col2 = st.columns(2)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mae = mean_absolute_error(actuals, preds)
    col1.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
    col2.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")

    # プロット
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(res_df['Date'], res_df['Actual Sales'], marker='o', label='Actual', color='gray', alpha=0.6)
    ax.plot(res_df['Date'], res_df['Predicted Sales'], marker='o', label='Predicted', color='blue', linewidth=2)
    ax.set_title("Sales Forecast vs Actual")
    ax.set_ylabel("Sales Quantity")
    ax.legend()
    st.pyplot(fig)

    st.dataframe(res_df[['Date', 'Actual Sales', 'Predicted Sales']].style.format("{:.1f}"))

# --- Tab 2: シミュレーション (What-If) ---
with tab2:
    st.subheader(" What-If Analysis: Adjust Input Factors")
    st.markdown("予測基準日時点での**「過去の実績」が変わっていたら、未来の予測はどう変わるか？**をシミュレーションします。")

    # シミュレーション用の入力データコピー
    sim_input = input_data.copy()

    col_sim1, col_sim2 = st.columns([1, 2])
    
    with col_sim1:
        st.markdown("### Parameters")
        
        # 1. 直近の売上 (lag_0)
        current_lag0 = float(sim_input['lag_0'].values[0])
        new_lag0 = st.slider(
            "Lag 0 (Sales on Base Date)", 
            min_value=0.0, max_value=current_lag0 * 2 + 10, 
            value=current_lag0, step=1.0
        )
        sim_input['lag_0'] = new_lag0
        
        # 2. 1週間前の売上 (lag_7)
        current_lag7 = float(sim_input['lag_7'].values[0])
        new_lag7 = st.slider(
            "Lag 7 (Sales 1 week ago)", 
            min_value=0.0, max_value=current_lag7 * 2 + 10, 
            value=current_lag7, step=1.0
        )
        sim_input['lag_7'] = new_lag7

        # 3. 直近7日間の平均 (roll_mean_7)
        # Note: 本来はlagが変わればrollingも再計算すべきですが、簡易的に独立して動かせるようにします
        current_roll7 = float(sim_input['roll_mean_7'].values[0])
        new_roll7 = st.slider(
            "Rolling Mean (Last 7 days avg)", 
            min_value=0.0, max_value=current_roll7 * 2 + 10, 
            value=current_roll7, step=0.5
        )
        sim_input['roll_mean_7'] = new_roll7
        
        if st.button("Reset Parameters"):
            st.rerun()

    with col_sim2:
        # 再予測
        sim_preds = []
        for target in target_cols:
            val = models[target].predict(sim_input)[0]
            sim_preds.append(val)
        
        # プロット比較
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # 元の予測
        ax2.plot(res_df['Date'], res_df['Predicted Sales'], label='Original Prediction', linestyle='--', color='gray')
        # シミュレーション予測
        ax2.plot(res_df['Date'], sim_preds, label='Simulated Prediction', marker='o', color='red', linewidth=2)
        
        ax2.set_title("Simulation Result: Impact on Forecast Curve")
        ax2.set_ylabel("Sales Quantity")
        ax2.legend()
        st.pyplot(fig2)
        
        # 差分の表示
        total_diff = sum(sim_preds) - sum(preds)
        st.info(f"Total Sales Difference (22 days): {total_diff:+.1f} units")

# --- Tab 3: 特徴量重要度 ---
with tab3:
    st.subheader("Feature Importance Analysis")
    
    # モデル選択
    selected_horizon = st.selectbox("Select Forecast Horizon", target_cols, index=0)
    
    model = models[selected_horizon]
    importance = model.feature_importance(importance_type='gain')
    feature_name = model.feature_name()
    
    # DataFrame化してソート
    imp_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
    imp_df = imp_df.sort_values('importance', ascending=False).head(15)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=imp_df, ax=ax3, palette='viridis')
    ax3.set_title(f"Feature Importance for {selected_horizon}")
    st.pyplot(fig3)