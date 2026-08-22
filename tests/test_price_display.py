"""
ATLAS — 저단가 코인 가격 표시
=============================
SHIBUSDT(진입가 6.12e-06) 포지션에서 실제로 터진 결함을 고정한다.

대시보드 API가 `round(entry, 4)`로 가격을 내보냈는데, 6.12e-06은 4자리
반올림하면 **정확히 0.0**이다. 그래서 entry/sl/tp가 전부 0으로 나갔고
UI가 미실현 손익을 `(현재가 - 0) x 수량`으로 계산해 **포지션 원금
$14.55를 통째로 수익으로 표시**했다. SL/TP 거리는 0%, 진행 바는 중앙
고정. 봇 로그·텔레그램도 `:,.4f` 때문에 "0.0000 → 0.0000"만 찍혀
체결가를 확인할 수 없었다.

가격은 자릿수가 심볼마다 6자리 이상 차이 난다(BTC 63,491 vs SHIB
0.00000612). 고정 소수점은 어느 한쪽을 반드시 망가뜨린다.
"""
import atlas_spot_main as sm
import atlas_web_dashboard as dash

# 실제 라이브 포지션에서 가져온 값들
SHIB_ENTRY = 6.12e-06
SHIB_SL    = 5.387142857142858e-06
SHIB_QTY   = 2382825.79
NEIRO_ENTRY = 0.000105717851126009


class TestDashboardRounding:
    def test_subcent_price_does_not_collapse_to_zero(self):
        """이 한 줄이 무너지면 대시보드가 원금을 수익으로 표시한다."""
        assert dash._round_px(SHIB_ENTRY) > 0, (
            'SHIB 진입가가 0으로 뭉개졌다 — UI가 (현재가-0)*수량을 '
            '미실현 손익으로 계산해 포지션 원금을 수익으로 표시한다')
        assert dash._round_px(SHIB_SL) > 0
        assert dash._round_px(NEIRO_ENTRY) > 0

    def test_keeps_six_significant_digits(self):
        for v in (SHIB_ENTRY, SHIB_SL, NEIRO_ENTRY, 0.0113, 0.08014):
            out = dash._round_px(v)
            assert abs(out - v) / v < 1e-5, f'{v} → {out} 유효숫자 손실'

    def test_normal_prices_unchanged(self):
        """기존 동작 보존 — 1달러 이상 가격은 종전대로 4자리."""
        assert dash._round_px(2.379) == 2.379
        assert dash._round_px(63491.0) == 63491.0
        assert dash._round_px(1.23456789) == 1.2346   # 1 이상은 4자리 유지

    def test_degenerate_inputs(self):
        for bad in (0, None, '', 'abc', float('nan'), float('inf')):
            assert dash._round_px(bad) == 0.0


class TestBotPriceFormat:
    def test_subcent_price_is_readable(self):
        """로그·텔레그램에서 체결가를 읽을 수 있어야 한다."""
        out = sm._fmt_px(SHIB_ENTRY)
        assert out not in ('0', '0.0000'), (
            f'SHIB 체결가가 {out}으로 찍힌다 — 매수/청산 알림에서 '
            f'가격을 확인할 수 없다')
        assert float(out.replace(',', '')) > 0

    def test_precision_scales_with_magnitude(self):
        assert sm._fmt_px(63491.0) == '63,491.0000'   # 큰 가격은 종전과 동일
        assert sm._fmt_px(2.379) == '2.3790'
        for v in (SHIB_ENTRY, NEIRO_ENTRY, 0.0113):
            assert abs(float(sm._fmt_px(v).replace(',', '')) - v) / v < 1e-3

    def test_degenerate_inputs(self):
        for bad in (0, None, 'abc'):
            assert sm._fmt_px(bad) == '0'


class TestUnrealizedPnlIsNotInflated:
    """대시보드 UI가 쓰는 식을 파이썬으로 재현해 회귀를 막는다.

    UI: upnl = (cur - entry) * qty   (entry가 0이면 원금 전체가 수익)
    """

    def test_pnl_uses_real_entry(self):
        cur = 6.05e-06
        entry = dash._round_px(SHIB_ENTRY)
        upnl = (cur - entry) * SHIB_QTY
        assert abs(upnl) < 1.0, (
            f'미실현 손익 ${upnl:.2f} — 진입가가 0으로 뭉개지면 '
            f'${cur * SHIB_QTY:.2f}(원금 전체)가 수익으로 찍힌다')
        assert upnl < 0, '진입가보다 현재가가 낮으므로 손실이어야 한다'


class TestFixedRoundingIsNotReintroduced:
    """소스 수준 래칫 — 가격에 고정 4자리 반올림이 다시 들어오는 것을 막는다.

    이 결함은 `round(entry, 4)` 한 줄에서 나왔고, 값이 0이 된 뒤로는
    UI가 조용히 원금을 수익으로 표시했다(예외도, 로그도 없었다).
    같은 패턴이 다시 들어오면 여기서 잡는다.
    """

    def test_position_payload_has_no_fixed_price_rounding(self):
        from pathlib import Path
        src = Path(dash.__file__).read_text(encoding='utf-8')
        for pattern in ("round(entry, 4)", "round(sl, 4)",
                        "round(float(p['tp']), 4)",
                        "round(float(r['entry_price']), 4)",
                        "round(float(r['exit_price']), 4)"):
            assert pattern not in src, (
                f'{pattern} — 저단가 코인(SHIB 6.12e-06)이 0으로 뭉개진다. '
                f'_round_px()를 쓸 것')
