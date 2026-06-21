#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
단일종목 MAC-IRL — 기본(base) 데이터 수집  (FinanceDataReader 버전)
대상: 삼성전자(005930), 일별, 2021-01-01 ~ 2025-12-31

수집(투자자 순매수 제외):
  date, close, return, trading_value, usdkrw, kospi200_return
  * 투자자별(외국인/기관/개인) 순매수는 KRX 사이트에서 따로 받아 나중에 합칩니다.

소스: FinanceDataReader (네이버 등) — KRX 로그인 불필요!
  - 005930  : 삼성전자 OHLCV
  - KS200   : KOSPI200 지수
  - USD/KRW : 원/달러 환율

설치:  pip3 install finance-datareader pandas openpyxl
실행:  python3 collect_base.py
출력:  samsung_base_2021_2025.csv
"""

import sys
import time
import pandas as pd
import FinanceDataReader as fdr

TICKER = "005930"
START = "2021-01-01"
END   = "2025-12-31"
DL_START = "2020-12-01"   # 수익률 계산용 버퍼
OUT_CSV  = "samsung_base_2021_2025.csv"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pick_col(df, keywords):
    for c in df.columns:
        for k in keywords:
            if k.lower() in str(c).lower():
                return c
    raise KeyError(f"컬럼 못 찾음. 키워드={keywords}, 실제={list(df.columns)}")


def get_ohlcv():
    log("삼성전자(005930) OHLCV 수집...")
    df = fdr.DataReader(TICKER, DL_START, END)
    if df is None or df.empty:
        raise RuntimeError("삼성전자 시세가 비었습니다. 네트워크를 확인하세요.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    close_col = pick_col(df, ["Close"])
    out = pd.DataFrame(index=df.index)
    out["close"] = df[close_col]
    # 거래대금: 'Amount' 컬럼이 있으면 사용, 없으면 종가×거래량 근사
    try:
        out["trading_value"] = df[pick_col(df, ["Amount"])]
    except KeyError:
        out["trading_value"] = df[close_col] * df[pick_col(df, ["Volume"])]
        log("주의: 거래대금(Amount)이 없어 '종가×거래량'으로 근사했습니다.")
    out["return"] = out["close"].pct_change()
    return out


def get_kospi200():
    log("KOSPI200(KS200) 지수 수집...")
    df = fdr.DataReader("KS200", DL_START, END)
    if df is None or df.empty:
        raise RuntimeError("KOSPI200가 비었습니다.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    out = pd.DataFrame(index=df.index)
    out["kospi200_return"] = df[pick_col(df, ["Close"])].pct_change()
    return out


def get_usdkrw():
    log("USD/KRW 환율 수집...")
    df = fdr.DataReader("USD/KRW", DL_START, END)
    if df is None or df.empty:
        raise RuntimeError("환율이 비었습니다.")
    df = df.copy()
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    out = pd.DataFrame(index=df.index)
    out["usdkrw"] = df[pick_col(df, ["Close"])]
    return out


def main():
    ohlcv = get_ohlcv()
    k200  = get_kospi200()
    fx    = get_usdkrw()

    df = ohlcv.join(k200, how="left").join(fx, how="left")
    df["usdkrw"] = df["usdkrw"].ffill()
    df["kospi200_return"] = df["kospi200_return"].ffill()
    df = df[(df.index >= pd.to_datetime(START)) & (df.index <= pd.to_datetime(END))]

    cols = ["close", "return", "trading_value", "usdkrw", "kospi200_return"]
    df = df.reset_index()[["date"] + cols]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    log(f"저장 완료: {OUT_CSV} (행 {len(df)}, 열 {df.shape[1]})")
    print("\n=== 앞 3행 ===")
    print(df.head(3).to_string(index=False))
    print("\n=== 뒤 3행 ===")
    print(df.tail(3).to_string(index=False))
    print("\n=== 결측치 수 ===")
    print(df.isna().sum().to_string())
    print(f"\n기간: {df['date'].min()} ~ {df['date'].max()}, 거래일 수: {len(df)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"오류: {e}")
        sys.exit(1)
