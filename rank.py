import datetime
from glob import glob
import numpy as np
import pandas as pd
import streamlit as st

filepaths = glob('data2/*.csv')

df = pd.DataFrame({
    'date': ['選択してください', '2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06', '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12',
            '2019-01', '2019-02', '2019-03', '2019-04', '2019-05', '2019-06', '2019-07', '2019-08', '2019-09', '2019-10', '2019-11', '2019-12',
            '2020-01', '2020-02', '2020-03', '2020-04', '2020-05', '2020-06', '2020-07', '2020-08', '2020-09', '2020-10', '2020-11', '2020-12',
            '2021-01', '2021-02', '2021-03', '2021-04', '2021-05'],
    'fileNo': [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,
            30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40],
})

item_list = df['date'].unique()

st.title('会員ランク算出')

st.markdown(rf'''
<br>
''', unsafe_allow_html=True)

col10, col11 = st.beta_columns(2)
with col10:
    start_date = st.selectbox(
        '開始月を選択：',
        item_list
    )
with col11:
    month = st.selectbox(
        '何か月分のデータで算出しますか？',
        (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12))

st.sidebar.write('会員ランク算定金額設定')
diamond_rank = st.sidebar.number_input('Diamond',  min_value=0, max_value=1000000, step=1000, value=100000)
platinum_rank = st.sidebar.number_input('Platinum',  min_value=0, max_value=1000000, step=1000, value=50000)         
gold_rank = st.sidebar.number_input('Gold',  min_value=0, max_value=1000000, step=1000, value=12000)
silver_rank = st.sidebar.number_input('Silver',  min_value=0, max_value=1000000, step=500, value=5000)
bronze_rank = st.sidebar.number_input('Bronze',  min_value=0, max_value=1000000, step=1, value=1)

st.sidebar.markdown(rf'''
<br>
''', unsafe_allow_html=True)

st.sidebar.write('ポイント倍率設定')
diamond_x = st.sidebar.number_input('Diamond',  min_value=0, max_value=100, step=1, value=5)
platinum_x = st.sidebar.number_input('Platinum',  min_value=0, max_value=100, step=1, value=4)         
gold_x = st.sidebar.number_input('Gold',  min_value=0, max_value=100, step=1, value=2)
silver_x = st.sidebar.number_input('Silver',  min_value=0, max_value=100, step=1, value=1)
bronze_x = st.sidebar.number_input('Bronze',  min_value=0, max_value=100, step=1, value=1)

st.markdown(rf'''
<br>
''', unsafe_allow_html=True)

if start_date != '選択してください':
    st.write('データ期間：', start_date, 'から', month, 'ヵ月間を集計')

st.markdown(rf'''
<br>
''', unsafe_allow_html=True)

df_filter = df[df['date']== start_date]
df_fileNo = int(df_filter['fileNo'])

if df_fileNo >= 0:
    if month > 0:
        data_list = []
        for file in filepaths[df_fileNo: df_fileNo + month]:
            data_list.append(pd.read_csv(file))

        df = pd.concat(data_list)
        df['税抜受注金額'] = df['税抜受注金額'].replace(',', '', regex=True).astype(int)
        df = df[(df['受注方法コード'] == 10) & df['受注売上完了区分'] == 1]

        df_ = df.pivot_table(index='得意先コード', columns='受注方法コード', values='税抜受注金額', aggfunc='sum')
        df_ = df_.replace(np.nan, 0, regex=True)
        df_ = df_.reset_index()
        df_sum = df_[10].sum()
        df_len = len(df_)

        # 利用回数カウント
        df_2 = df.drop_duplicates(['受注NO'])
        df_2 = df_2.pivot_table(index='得意先コード', columns='受注方法コード', values='税抜受注金額', aggfunc='count')
        df_2_mean = round(df_2[10].mean(), 2)
        df_3 = pd.merge(df_2, df_, how='left', on='得意先コード')

        df_mean2 = round(df_3['10_y'] / df_3['10_x'], 2)
        df_mean2 = round(df_mean2.mean(), 2)

        # ブロンズ算出
        df_bronze = df_[(df_[10] >= bronze_rank) & (df_[10] < silver_rank)]
        df_bronze_sum = df_bronze[10].sum()
        df_bronze_sum2 = len(df_bronze)
        df_bronze_sum_per = round((df_bronze_sum / df_sum) * 100, 2)
        df_bronze_sum_per2 = round((df_bronze_sum2 / df_len) * 100, 2)
        df_bronze_3 = df_3[(df_3['10_y'] >= bronze_rank) & (df_3['10_y'] < silver_rank)]
        df_bronze_mean = round(df_bronze_3['10_x'].mean(), 2)

        df_bronze_mean2 = round(df_bronze_sum / (df_bronze_mean * df_bronze_sum2), 2)

        # シルバー算出
        df_silver = df_[(df_[10] >= silver_rank) & (df_[10] < gold_rank)]
        df_silver_sum = df_silver[10].sum()
        df_silver_sum2 = len(df_silver)
        df_silver_sum_per = round((df_silver_sum / df_sum) * 100, 2)
        df_silver_sum_per2 = round((df_silver_sum2 / df_len) * 100, 2)
        df_silver_3 = df_3[(df_3['10_y'] >= silver_rank) & (df_3['10_y'] < gold_rank)]
        df_silver_mean = round(df_silver_3['10_x'].mean(), 2)

        df_silver_mean2 = round(df_silver_sum / (df_silver_mean * df_silver_sum2), 2)

        # ゴールド算出
        df_gold = df_[(df_[10] >= gold_rank) & (df_[10] < platinum_rank)]
        df_gold_sum = df_gold[10].sum()
        df_gold_sum2 = len(df_gold)
        df_gold_sum_per = round((df_gold_sum / df_sum) * 100, 2)
        df_gold_sum_per2 = round((df_gold_sum2 / df_len) * 100, 2)
        df_gold_3 = df_3[(df_3['10_y'] >= gold_rank) & (df_3['10_y'] < platinum_rank)]
        df_gold_mean = round(df_gold_3['10_x'].mean(), 2)

        df_gold_mean2 = round(df_gold_sum / (df_gold_mean * df_gold_sum2), 2)

        # プラチナ算出
        df_platinum = df_[(df_[10] >= platinum_rank) & (df_[10] < diamond_rank)]
        df_platinum_sum = df_platinum[10].sum()
        df_platinum_sum2 = len(df_platinum)
        df_platinum_sum_per = round((df_platinum_sum / df_sum) * 100, 2)
        df_platinum_sum_per2 = round((df_platinum_sum2 / df_len) * 100, 2)
        df_platinum_3 = df_3[(df_3['10_y'] >= platinum_rank) & (df_3['10_y'] < diamond_rank)]
        df_platinum_mean = round(df_platinum_3['10_x'].mean(), 2)

        df_platinum_mean2 = round(df_platinum_sum / (df_platinum_mean * df_platinum_sum2), 2)

        # ダイアモンド算出
        df_diamond = df_[(df_[10] >= diamond_rank)]
        df_diamond_sum = df_diamond[10].sum()
        df_diamond_sum2 = len(df_diamond)
        df_diamond_sum_per = round((df_diamond_sum / df_sum) * 100, 2)
        df_diamond_sum_per2 = round((df_diamond_sum2 / df_len) * 100, 2)
        df_diamond_3 = df_3[(df_3['10_y'] >= diamond_rank)]
        df_diamond_mean = round(df_diamond_3['10_x'].mean(), 2)

        df_diamond_mean2 = round(df_diamond_sum / (df_diamond_mean * df_diamond_sum2), 2)

        # 合計
        df_total_sum2 = df_diamond_sum2 + df_platinum_sum2 + df_gold_sum2 + df_silver_sum2 + df_bronze_sum2
        df_total_sum = df_diamond_sum + df_platinum_sum + df_gold_sum + df_silver_sum + df_bronze_sum
        df_total_mean = df_diamond_mean + df_platinum_mean + df_gold_mean + df_silver_mean + df_bronze_mean
        df_total_mean2 = df_diamond_mean2 + df_platinum_mean2 + df_gold_mean2 + df_silver_mean2 + df_bronze_mean2
        df_totla_sum_per2 = 100

        # データフレームで可視化する
        st.write('NEWランク運用時シミュレーション')
        st.table(pd.DataFrame({
            'ランク': ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze', 'Total'],
            '査定': ["{:,}".format(diamond_rank), "{:,}".format(platinum_rank), "{:,}".format(gold_rank), "{:,}".format(silver_rank), "{:,}".format(bronze_rank), ''],
            '会員数': ["{:,}".format(df_diamond_sum2), "{:,}".format(df_platinum_sum2), "{:,}".format(df_gold_sum2), "{:,}".format(df_silver_sum2), "{:,}".format(df_bronze_sum2), "{:,}".format(df_total_sum2)],
            '会員比': ["{:,}".format(df_diamond_sum_per2), "{:,}".format(df_platinum_sum_per2), "{:,}".format(df_gold_sum_per2), "{:,}".format(df_silver_sum_per2), "{:,}".format(df_bronze_sum_per2), "{:,}".format(df_totla_sum_per2)],
            '合計額': ["{:,}".format(df_diamond_sum), "{:,}".format(df_platinum_sum), "{:,}".format(df_gold_sum), "{:,}".format(df_silver_sum), "{:,}".format(df_bronze_sum), "{:,}".format(df_total_sum)],
            '平均回数': ["{:,}".format(df_diamond_mean), "{:,}".format(df_platinum_mean), "{:,}".format(df_gold_mean), "{:,}".format(df_silver_mean), "{:,}".format(df_bronze_mean), "{:,}".format(df_2_mean)],
            '平均額/回': ["{:,}".format(df_diamond_mean2), "{:,}".format(df_platinum_mean2), "{:,}".format(df_gold_mean2), "{:,}".format(df_silver_mean2), "{:,}".format(df_bronze_mean2), "{:,}".format(df_mean2)],
            '平均額/年': ["{:,}".format(round(df_diamond_mean2 * df_diamond_mean),0), "{:,}".format(round(df_platinum_mean2 * df_platinum_mean),0), "{:,}".format(round(df_gold_mean2 * df_gold_mean),0), "{:,}".format(round(df_silver_mean2 * df_silver_mean),0), "{:,}".format(round(df_bronze_mean2 * df_bronze_mean),0), "{:,}".format(round(df_mean2 * df_2_mean),0)]
        }))

        # 現シルバー算出
        df_silver_ = df_[(df_[10] >= 3000) & (df_[10] < 6000)]
        df_silver_sum_ = df_silver_[10].sum()
        df_silver_sum2_ = len(df_silver_)
        df_silver_sum_per_ = round((df_silver_sum_ / df_sum) * 100, 2)
        df_silver_sum_per2_ = round((df_silver_sum2_ / df_len) * 100, 2)
        df_silver_3_ = df_3[(df_3['10_y'] >= 3000) & (df_3['10_y'] < 6000)]
        df_silver_mean_ = round(df_silver_3_['10_x'].mean(), 2)

        df_silver_mean2_ = round(df_silver_sum_ / (df_silver_mean_ * df_silver_sum2_), 2)

        # 現ゴールド算出
        df_gold_ = df_[(df_[10] >= 6000) & (df_[10] < 12000)]
        df_gold_sum_ = df_gold_[10].sum()
        df_gold_sum2_ = len(df_gold_)
        df_gold_sum_per_ = round((df_gold_sum_ / df_sum) * 100, 2)
        df_gold_sum_per2_ = round((df_gold_sum2_ / df_len) * 100, 2)
        df_gold_3_ = df_3[(df_3['10_y'] >= 6000) & (df_3['10_y'] < 120000)]
        df_gold_mean_ = round(df_gold_3_['10_x'].mean(), 2)

        df_gold_mean2_ = round(df_gold_sum_ / (df_gold_mean_ * df_gold_sum2_), 2)

        # 現プラチナ算出
        df_platinum_ = df_[(df_[10] >= 12000)]
        df_platinum_sum_ = df_platinum_[10].sum()
        df_platinum_sum2_ = len(df_platinum_)
        df_platinum_sum_per_ = round((df_platinum_sum_ / df_sum) * 100, 2)
        df_platinum_sum_per2_ = round((df_platinum_sum2_ / df_len) * 100, 2)
        df_platinum_3_ = df_3[(df_3['10_y'] >= 12000)]
        df_platinum_mean_ = round(df_platinum_3_['10_x'].mean(), 2)

        df_platinum_mean2_ = round(df_platinum_sum_ / (df_platinum_mean_ * df_platinum_sum2_), 2)

        # 現通常算出
        df_normal_ = df_[(df_[10] < 3000) & (df_[10] >= 1)]
        df_normal_sum_ = df_normal_[10].sum()
        df_normal_sum2_ = len(df_normal_)
        df_normal_sum_per_ = round((df_normal_sum_ / df_sum) * 100, 2)
        df_normal_sum_per2_ = round((df_normal_sum2_ / df_len) * 100, 2)
        df_normal_3_ = df_3[(df_3['10_y'] < 3000) & (df_3['10_y'] >= 1)]
        df_normal_mean_ = round(df_normal_3_['10_x'].mean(), 2)

        df_normal_mean2_ = round(df_normal_sum_ / (df_normal_mean_ * df_normal_sum2_), 2)

        # 合計
        df_total_sum2_ = df_platinum_sum2_ + df_gold_sum2_ + df_silver_sum2_ + df_normal_sum2_
        df_total_sum_ = df_platinum_sum_ + df_gold_sum_ + df_silver_sum_ + df_normal_sum_
        df_total_mean_ = df_platinum_mean_ + df_gold_mean_ + df_silver_mean_ + df_normal_mean_
        df_total_mean2_ = df_platinum_mean2_ + df_gold_mean2_ + df_silver_mean2_ + df_normal_mean2_
        df_totla_sum_per2_ = 100

        # データフレームで可視化する
        st.write('現ランク運用時シミュレーション')
        st.table(pd.DataFrame({
            'ランク': ['Platinum', 'Gold', 'Silver', 'Normal', 'Total'],
            '査定': ['12,000', '6,000', '3,000', '0', ''],
            '会員数': ["{:,}".format(df_platinum_sum2_), "{:,}".format(df_gold_sum2_), "{:,}".format(df_silver_sum2_), "{:,}".format(df_normal_sum2_), "{:,}".format(df_total_sum2_)],
            '会員比': ["{:,}".format(df_platinum_sum_per2_), "{:,}".format(df_gold_sum_per2_), "{:,}".format(df_silver_sum_per2_), "{:,}".format(df_normal_sum_per2_), "{:,}".format(df_totla_sum_per2_)],
            '合計額': ["{:,}".format(df_platinum_sum_), "{:,}".format(df_gold_sum_), "{:,}".format(df_silver_sum_), "{:,}".format(df_normal_sum_), "{:,}".format(df_total_sum_)],
            '平均回数': ["{:,}".format(df_platinum_mean_), "{:,}".format(df_gold_mean_), "{:,}".format(df_silver_mean_), "{:,}".format(df_normal_mean_), "{:,}".format(df_2_mean)],
            '平均額/回': ["{:,}".format(df_platinum_mean2_), "{:,}".format(df_gold_mean2_), "{:,}".format(df_silver_mean2_), "{:,}".format(df_normal_mean2_), "{:,}".format(df_mean2)],
            '平均額/年': ["{:,}".format(round(df_platinum_mean2_ * df_platinum_mean_),0), "{:,}".format(round(df_gold_mean2_ * df_gold_mean_),0), "{:,}".format(round(df_silver_mean2_ * df_silver_mean_),0), "{:,}".format(round(df_normal_mean2_ * df_normal_mean_),0), "{:,}".format(round(df_mean2 * df_2_mean),0)]
        }))

        # ポイント計算
        df_diamond_p = round((df_diamond_sum * (diamond_x / 100)), 0)
        df_platinum_p = round((df_platinum_sum * (platinum_x / 100)), 0)
        df_gold_p = round((df_gold_sum * (gold_x / 100)), 0)
        df_silver_p = round((df_silver_sum * (silver_x / 100)), 0)
        df_bronze_p = round((df_bronze_sum * (bronze_x / 100)), 0)
        df_total_p = df_diamond_p + df_platinum_p + df_gold_p + df_silver_p + df_bronze_p

        # 現運用ポイント計算
        df_diamond_p_ = round((df_diamond_sum * 0.01), 0)
        df_platinum_p_ = round((df_platinum_sum * 0.01), 0)
        df_gold_p_ = round((df_gold_sum * 0.01), 0)
        df_silver_p_ = round((df_silver_sum * 0.01), 0)
        df_bronze_p_ = round((df_bronze_sum * 0.01), 0)
        df_total_p_ = df_diamond_p_ + df_platinum_p_ + df_gold_p_ + df_silver_p_ + df_bronze_p_

        # ポイント差異
        df_diamond_p_re = df_diamond_p - df_diamond_p_
        df_platinum_p_re = df_platinum_p - df_platinum_p_
        df_gold_p_re = df_gold_p - df_gold_p_
        df_silver_p_re = df_silver_p - df_silver_p_
        df_bronze_p_re = df_bronze_p - df_bronze_p_
        df_total_p_re = df_diamond_p_re + df_platinum_p_re + df_gold_p_re + df_silver_p_re + df_bronze_p_re

        # 全体のポイント付与率
        df_total_p_per = round((df_total_p / df_total_sum) *100, 2)

        # ポイント発行シミュレーション
        st.write('ポイント発行シミュレーション')
        st.table(pd.DataFrame({
            'ランク': ['Diamond', 'Platinum', 'Gold', 'Silver', 'Bronze', 'Total'],
            'P付与率': ["{:,}".format(diamond_x), "{:,}".format(platinum_x), "{:,}".format(gold_x), "{:,}".format(silver_x), "{:,}".format(bronze_x), "{:,}".format(df_total_p_per)],
            '付与対象額': ["{:,}".format(df_diamond_sum), "{:,}".format(df_platinum_sum), "{:,}".format(df_gold_sum), "{:,}".format(df_silver_sum), "{:,}".format(df_bronze_sum), "{:,}".format(df_total_sum)],
            '付与P': ["{:,}".format(df_diamond_p), "{:,}".format(df_platinum_p), "{:,}".format(df_gold_p), "{:,}".format(df_silver_p), "{:,}".format(df_bronze_p), "{:,}".format(df_total_p)],
            '現運用P': ["{:,}".format(df_diamond_p_), "{:,}".format(df_platinum_p_), "{:,}".format(df_gold_p_), "{:,}".format(df_silver_p_), "{:,}".format(df_bronze_p_), "{:,}".format(df_total_p_)],
            'P増減': ["{:,}".format(df_diamond_p_re), "{:,}".format(df_platinum_p_re), "{:,}".format(df_gold_p_re), "{:,}".format(df_silver_p_re), "{:,}".format(df_bronze_p_re), "{:,}".format(df_total_p_re)],
        }))