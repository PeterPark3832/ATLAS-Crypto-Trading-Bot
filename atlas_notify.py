"""
ATLAS — 배치·대시보드용 동기 텔레그램 전송 (단일 출처)
=====================================================
같은 sendMessage POST가 4개 파일에 복사돼 있었다(weekly/monthly/reoptimize
의 `tg()` + 대시보드의 `_tg()`). monthly 사본의 docstring이 "weekly_report.tg
와 동일 규약"이라고 중복을 자인할 정도였다. 여기로 모은다.

⚠️ `atlas_spot_main._tg` 는 통합 대상이 **아니다.** 데몬은 큐 기반
   비동기(전송 실패가 매매 루프를 절대 막으면 안 됨)이고, 배치 스크립트는
   동기 + print 폴백(cron 출력이 유실되면 안 됨)이다. 계약이 다르다.

토큰을 모듈 전역이 아니라 **호출 인자로** 받는 이유:
  reoptimize/monthly 의 `--no-tg` 는 `global TG_TOKEN = ''` 로 구현돼 있다.
  각 스크립트가 자기 전역을 호출 시점에 읽어 넘기면 그 메커니즘이
  그대로 보존된다. 이 모듈이 자체 전역을 가지면 --no-tg가 조용히 죽는다.
"""

import logging

import requests


def send_telegram(text: str, token: str, chat_id: str, *,
                  timeout: float = 10,
                  print_fallback: bool = False,
                  reprint_on_error: bool = False,
                  logger: logging.Logger | None = None) -> bool:
    """텔레그램 sendMessage. 성공 시 True.

    기존 4개 사본의 서로 다른 계약을 파라미터로 보존한다:
      · weekly     : timeout=10, print_fallback=True   (실패 시 오류만 출력)
      · monthly/reopt: timeout=15, print_fallback=True, reprint_on_error=True
                       (실패 시 본문도 출력 — cron 로그에 리포트가 남게)
      · dashboard  : timeout=8,  logger=log            (print 대신 log.error)

    자격증명이 없으면 print_fallback 여부에 따라 본문을 stdout으로 흘리고
    False를 돌려준다 — 배치 스크립트의 출력 유실 방지 계약이다.
    """
    if not token or not chat_id:
        if print_fallback:
            print(text)
        return False
    try:
        requests.post(
            f'https://api.telegram.org/bot{token}/sendMessage',
            data={'chat_id': chat_id, 'text': text},
            timeout=timeout,
        )
        return True
    except Exception as e:
        err = f'TG 전송 실패: {e}'
        if logger is not None:
            logger.error(err)
        else:
            print(err)
        if reprint_on_error:
            print(text)
        return False
