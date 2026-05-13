---
name: 股票可视化分析
description: 基于K线数据对股票进行可视化分析，绘制日K、周K图形，并给出买入卖出时机建议
---

# 主要功能

## 1. K线可视化
- `plot_daily_kline` - 绘制股票的日K线图
- `plot_weekly_kline` - 绘制股票的周K线图
- 支持前复权/后复权/不复权

## 2. 波动分析
- `analyze_daily_volatility` - 分析日波动幅度
- `analyze_weekly_volatility` - 分析周波动幅度
- 计算涨跌幅、成交量变化

## 3. 买卖建议
- `get_buy_sell_suggestion` - 综合技术分析给出买卖建议

| 信号类型 | 说明 | 操作建议 |
|----------|------|----------|
| 日线底部信号 | 长下影线、阳包阴、启明星 | 考虑买入 |
| 日线顶部信号 | 长上影线、阴包阳、黄昏星 | 考虑卖出 |
| 周线支撑位 | 回调到20周均线 | 买入或加仓 |
| 周线压力位 | 多次在某价格遇阻 | 减仓或观望 |
| 突破信号 | 放量阳线突破颈线 | 买入 |
| 止损位 | 最近标志性阳线最低点 | 设置止损 |

# 调用方法
```python
TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1/stock"

import requests
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import Annotated, Optional, Dict, List
import traceback
import io
import base64

# ========== 基础数据接口 ==========

def get_all_stock_code(keyword: Optional[str] = None) -> Dict:
    """查询所有股票，支持代码/名称模糊搜索"""
    url = f"{BASE_URL}/all?token={TOKEN}"
    if keyword:
        url += f"&keyWord={keyword}"
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

def get_day_line(code: str, startDate: Optional[str] = None, endDate: Optional[str] = None, type: int = 1) -> Dict:
    """获取日K线数据"""
    url = f"{BASE_URL}/kline/day?token={TOKEN}"
    params = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

def get_week_line(code: str, startDate: Optional[str] = None, endDate: Optional[str] = None, type: int = 1) -> Dict:
    """获取周K线数据"""
    url = f"{BASE_URL}/kline/week?token={TOKEN}"
    params = {"code": code, "startDate": startDate, "endDate": endDate, "type": type}
    try:
        response = requests.get(url, params=params, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

def get_stock_info(code: str) -> Dict:
    """获取股票基础信息"""
    url = f"{BASE_URL}?token={TOKEN}&code={code}"
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

def get_stock_minute_data(code: str) -> Dict:
    """获取分时数据"""
    url = f"{BASE_URL}/min?token={TOKEN}&code={code}"
    try:
        response = requests.get(url, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

# ========== K线绘图 ==========

def parse_kline_data(data: Dict) -> List[Dict]:
    """解析K线数据"""
    if data.get("code") != 0 or "data" not in data:
        return []
    return data["data"]

def plot_daily_kline(code: str, days: int = 60, adj_type: int = 1) -> str:
    """
    绘制日K线图
    :param code: 股票代码
    :param days: 显示天数
    :param adj_type: 0不复权, 1前复权, 2后复权
    :return: base64编码的图片
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

    data = get_day_line(code, start_date, end_date, adj_type)
    klines = parse_kline_data(data)

    if not klines:
        return ""

    # 取最近days天
    klines = klines[-days:]

    dates = []
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for k in klines:
        dates.append(datetime.strptime(k["date"], "%Y-%m-%d"))
        opens.append(float(k["open"]))
        highs.append(float(k["high"]))
        lows.append(float(k["low"]))
        closes.append(float(k["close"]))
        volumes.append(int(k["volume"]))

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"{code} 日K线图", fontsize=14)

    # 绘制K线
    for i in range(len(dates)):
        color = 'red' if closes[i] >= opens[i] else 'green'
        high_low = [ lows[i], highs[i]]
        open_close = [ opens[i], closes[i]]
        ax1.plot([dates[i], dates[i]], high_low, color=color, linewidth=0.8)
        ax1.plot([dates[i], dates[i]], open_close, color=color, linewidth=2)

    ax1.set_ylabel("价格")
    ax1.grid(True, alpha=0.3)

    # 绘制成交量
    colors = ['red' if closes[i] >= opens[i] else 'green' for i in range(len(dates))]
    ax2.bar(dates, volumes, color=colors, width=0.8)
    ax2.set_ylabel("成交量")
    ax2.set_xlabel("日期")
    ax2.grid(True, alpha=0.3)

    # 格式化x轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    # 转为base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64

def plot_weekly_kline(code: str, weeks: int = 52, adj_type: int = 1) -> str:
    """
    绘制周K线图
    :param code: 股票代码
    :param weeks: 显示周数
    :param adj_type: 0不复权, 1前复权, 2后复权
    :return: base64编码的图片
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(weeks=weeks * 2)).strftime("%Y-%m-%d")

    data = get_week_line(code, start_date, end_date, adj_type)
    klines = parse_kline_data(data)

    if not klines:
        return ""

    klines = klines[-weeks:]

    dates = []
    opens, highs, lows, closes, volumes = [], [], [], [], []

    for k in klines:
        dates.append(datetime.strptime(k["date"], "%Y-%m-%d"))
        opens.append(float(k["open"]))
        highs.append(float(k["high"]))
        lows.append(float(k["low"]))
        closes.append(float(k["close"]))
        volumes.append(int(k["volume"]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    fig.suptitle(f"{code} 周K线图", fontsize=14)

    for i in range(len(dates)):
        color = 'red' if closes[i] >= opens[i] else 'green'
        ax1.plot([dates[i], dates[i]], [lows[i], highs[i]], color=color, linewidth=0.8)
        ax1.plot([dates[i], dates[i]], [opens[i], closes[i]], color=color, linewidth=2)

    ax1.set_ylabel("价格")
    ax1.grid(True, alpha=0.3)

    colors = ['red' if closes[i] >= opens[i] else 'green' for i in range(len(dates))]
    ax2.bar(dates, volumes, color=colors, width=3)
    ax2.set_ylabel("成交量")
    ax2.set_xlabel("周")
    ax2.grid(True, alpha=0.3)

    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    return img_base64

# ========== 波动分析 ==========

def analyze_daily_volatility(code: str, days: int = 20) -> Dict:
    """分析日波动"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days * 2)).strftime("%Y-%m-%d")

    data = get_day_line(code, start_date, end_date, adj_type=1)
    klines = parse_kline_data(data)

    if len(klines) < days:
        return {}

    klines = klines[-days:]

    changes = []
    volumes = []
    for i in range(1, len(klines)):
        prev_close = float(klines[i - 1]["close"])
        curr_close = float(klines[i]["close"])
        change_pct = (curr_close - prev_close) / prev_close * 100
        changes.append(change_pct)
        volumes.append(int(klines[i]["volume"]))

    avg_change = sum(changes) / len(changes)
    max_up = max(changes)
    max_down = min(changes)
    avg_volume = sum(volumes) / len(volumes)

    return {
        "days": days,
        "avg_change_pct": round(avg_change, 2),
        "max_up_pct": round(max_up, 2),
        "max_down_pct": round(max_down, 2),
        "avg_volume": int(avg_volume),
        "recent_changes": [round(c, 2) for c in changes[-5:]]
    }

def analyze_weekly_volatility(code: str, weeks: int = 20) -> Dict:
    """分析周波动"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(weeks=weeks * 2)).strftime("%Y-%m-%d")

    data = get_week_line(code, start_date, end_date, adj_type=1)
    klines = parse_kline_data(data)

    if len(klines) < weeks:
        return {}

    klines = klines[-weeks:]

    changes = []
    for i in range(1, len(klines)):
        prev_close = float(klines[i - 1]["close"])
        curr_close = float(klines[i]["close"])
        change_pct = (curr_close - prev_close) / prev_close * 100
        changes.append(change_pct)

    avg_change = sum(changes) / len(changes)

    return {
        "weeks": weeks,
        "avg_change_pct": round(avg_change, 2),
        "max_up_pct": round(max(changes), 2),
        "max_down_pct": round(min(changes), 2),
        "recent_changes": [round(c, 2) for c in changes[-5:]]
    }

# ========== 买卖建议 ==========

def get_buy_sell_suggestion(code: str) -> Dict:
    """
    综合分析给出买入卖出建议
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date_weekly = (datetime.now() - timedelta(weeks=52)).strftime("%Y-%m-%d")
    start_date_daily = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")

    # 获取日K和周K数据
    daily_data = get_day_line(code, start_date_daily, end_date, adj_type=1)
    weekly_data = get_week_line(code, start_date_weekly, end_date, adj_type=1)

    daily_klines = parse_kline_data(daily_data)
    weekly_klines = parse_kline_data(weekly_data)

    if not daily_klines or not weekly_klines:
        return {"error": "无法获取数据"}

    # 最近60天日K用于信号分析
    recent_daily = daily_klines[-60:]
    # 最近20周周K用于趋势判断
    recent_weekly = weekly_klines[-20:]

    suggestions = []
    signals = []

    # ===== 日线信号分析 =====
    if len(recent_daily) >= 5:
        # 最近5天
        last_5 = recent_daily[-5:]

        # 底部信号检测
        for i in range(1, len(last_5)):
            prev = last_5[i - 1]
            curr = last_5[i]
            prev_close = float(prev["close"])
            curr_close = float(curr["close"])
            curr_low = float(curr["low"])
            curr_open = float(curr["open"])

            # 长下影线
            if curr_low < curr_open * 0.98 and curr_close > curr_open:
                signals.append({"type": "日线底部", "signal": "长下影线", "action": "买入", "date": curr["date"]})

            # 阳包阴
            if i > 0:
                prev_open = float(prev["open"])
                prev_close = float(prev["close"])
                if prev_close < prev_open and curr_close > curr_open and curr_close > prev_close and curr_open < prev_open:
                    signals.append({"type": "日线底部", "signal": "阳包阴", "action": "买入", "date": curr["date"]})

        # 顶部信号检测
        for i in range(1, len(last_5)):
            prev = last_5[i - 1]
            curr = last_5[i]
            curr_high = float(curr["high"])
            curr_open = float(curr["open"])
            curr_close = float(curr["close"])

            # 长上影线
            if curr_high > curr_open * 1.02 and curr_close < curr_open:
                signals.append({"type": "日线顶部", "signal": "长上影线", "action": "卖出", "date": curr["date"]})

    # ===== 周线趋势分析 =====
    if len(recent_weekly) >= 10:
        # 计算均线判断趋势
        week_closes = [float(k["close"]) for k in recent_weekly]
        ma5 = sum(week_closes[-5:]) / 5
        ma10 = sum(week_closes[-10:]) / 10
        ma20 = sum(week_closes[-20:]) / 20

        current_price = week_closes[-1]

        if ma5 > ma10 > ma20 and current_price > ma20:
            suggestions.append({"trend": "周线多头排列", "action": "买入", "reason": "上升趋势明确"})
        elif ma5 < ma10 < ma20 and current_price < ma20:
            suggestions.append({"trend": "周线空头排列", "action": "卖出/观望", "reason": "下降趋势明确"})
        else:
            suggestions.append({"trend": "周线震荡", "action": "观望/小仓位波段", "reason": "方向不明"})

        # 支撑压力位
        max_high = max([float(k["high"]) for k in recent_weekly[-10:]])
        min_low = min([float(k["low"]) for k in recent_weekly[-10:]])
        suggestions.append({"level": "近期压力位", "price": max_high, "action": "突破后买入"})
        suggestions.append({"level": "近期支撑位", "price": min_low, "action": "回调企稳买入"})

    # 综合建议
    buy_signals = [s for s in signals if s["action"] == "买入"]
    sell_signals = [s for s in signals if s["action"] == "卖出"]

    final_suggestion = "观望"
    if buy_signals and not sell_signals:
        final_suggestion = "可考虑买入"
    elif sell_signals and not buy_signals:
        final_suggestion = "建议减仓或卖出"
    elif buy_signals and sell_signals:
        final_suggestion = "注意高抛低吸"

    return {
        "code": code,
        "final_suggestion": final_suggestion,
        "trend_analysis": suggestions,
        "signals": signals,
        "daily_chart": plot_daily_kline(code, days=60),
        "weekly_chart": plot_weekly_kline(code, weeks=52)
    }
```
