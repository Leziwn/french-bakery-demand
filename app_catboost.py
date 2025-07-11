import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import urllib.request

# ✅ Streamlit Cloud에서도 동작 가능한 한글 폰트 설정
font_url = "https://github.com/google/fonts/raw/main/ofl/nanumgothic/NanumGothic-Regular.ttf"
font_path = "/tmp/NanumGothic.ttf"

if not os.path.exists(font_path):
    urllib.request.urlretrieve(font_url, font_path)

fm.fontManager.addfont(font_path)
font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc("font", family=font_name)
plt.rcParams["axes.unicode_minus"] = False


# ② 데이터 불러오기
df = pd.read_csv("article_predictions_final.csv", parse_dates=["date"])

# ③ 타이틀 및 사이드바
st.title("📊 실제 수요량 vs 예측 수요량")
selected_article = st.sidebar.selectbox("🔍 상품을 선택하세요", sorted(df["article"].unique()))

# ④ 필터링
filtered_df = df[df["article"] == selected_article].sort_values("date")

# 👉 한 달 2회만 추출 (15일 & 말일 기준)
filtered_df["day"] = filtered_df["date"].dt.day
plot_df = filtered_df[filtered_df["day"].isin([15, filtered_df["date"].dt.days_in_month])]

# ⑤ MAE, RMSE 계산
mae = mean_absolute_error(plot_df["actual"], plot_df["predicted"])
rmse = np.sqrt(mean_squared_error(plot_df["actual"], plot_df["predicted"]))

st.metric("📉 MAE (평균절대오차)", f"{mae:.2f}")
st.metric("📉 RMSE (평균제곱근오차)", f"{rmse:.2f}")

# ⑥ 시각화 (scatter plot)
st.subheader("📍 실제 수요량 vs 예측 수요량")

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(plot_df["date"], plot_df["actual"], label="실제 수요량", color="tab:blue", marker="o", s=60, alpha=0.7)
ax.scatter(plot_df["date"], plot_df["predicted"], label="예측 수요량", color="tab:orange", marker="x", s=60, alpha=0.7)

ax.set_title(f"{selected_article} - 예측 vs 실제", fontsize=14)
ax.set_xlabel("날짜")
ax.set_ylabel("수량")
ax.legend()
ax.grid(True, linestyle="--", alpha=0.3)
plt.xticks(rotation=45)

st.pyplot(fig)

# ⑦ 데이터 테이블 (📄 전체 데이터 보기)
st.subheader("📄 전체 수요 예측 데이터")
st.dataframe(filtered_df[["date", "actual", "predicted"]].reset_index(drop=True))
