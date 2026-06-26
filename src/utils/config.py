# 파이썬이 타입 힌트(아래 dict[str, Any] 같은 표기)를 더 유연하게 해석하게 해주는 한 줄.
# 동작에는 영향을 주지 않으니 "그냥 맨 위에 두는 관례"로 이해하면 됩니다.
from __future__ import annotations

# import = 다른 곳에 이미 만들어진 도구를 가져와 쓰는 것.
from copy import deepcopy   # deepcopy: 데이터를 "완전 복사"해서 원본과 분리시킴 (아래서 설명)
from pathlib import Path    # Path: 파일 경로를 다루는 편리한 객체 (문자열 경로보다 안전함)
from typing import Any      # Any: "아무 타입이나 올 수 있다"는 표시 (설정값은 숫자·문자·리스트 등 제각각이라)

import yaml                 # yaml: .yaml 설정 파일을 읽고 쓰는 외부 라이브러리


# ─────────────────────────────────────────────────────────────
# YAML 파일 하나를 읽어서 파이썬 dict(딕셔너리)로 돌려주는 함수.
# dict = {"키": 값} 형태의 자료구조. YAML의 "key: value"가 그대로 이렇게 변환됩니다.
#   path: str | Path  → 입력은 문자열 경로 또는 Path 객체 ('|'는 "둘 중 하나" 라는 뜻)
#   -> dict[str, Any] → 반환값은 "문자열 키 / 아무 값" 형태의 dict 라는 표시(힌트일 뿐, 강제는 아님)
# ─────────────────────────────────────────────────────────────
def load_yaml(path: str | Path) -> dict[str, Any]:
    # with ... as f: 파일을 열고, 블록이 끝나면 자동으로 닫아주는 안전한 방식.
    # "r" = 읽기 모드(read), encoding="utf-8" = 한글 등이 깨지지 않게 하는 설정.
    with Path(path).open("r", encoding="utf-8") as f:
        # safe_load: 파일 내용을 파이썬 자료(dict 등)로 변환.
        # "or {}" = 파일이 비어 있어서 결과가 None이면 빈 dict({})로 대체하라는 안전장치.
        loaded = yaml.safe_load(f) or {}
    # 읽은 결과가 dict가 아니면(예: 리스트만 적혀 있으면) 에러를 내서 일찍 알려줌.
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected mapping in config file: {path}")
    return loaded   # 정상이면 dict를 돌려줌


# ─────────────────────────────────────────────────────────────
# 두 dict를 "깊게(deep)" 병합하는 함수.
# 보통 dict를 합치면 같은 키일 때 통째로 덮어쓰지만,
# 여기서는 값이 또 dict면 그 안쪽까지 들어가서 세밀하게 합칩니다(=중첩 설정 보존).
#   base     = 바탕이 되는 설정
#   override = 위에 덮어쓸 설정 (같은 키가 있으면 이쪽이 이김)
# ─────────────────────────────────────────────────────────────
def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    # base를 직접 수정하면 호출한 쪽의 원본까지 바뀌어버림.
    # deepcopy로 완전 복사본을 만들어 그 복사본만 고쳐서 부작용을 막음.
    result = deepcopy(base)
    # override의 모든 (키, 값) 쌍을 하나씩 돌면서 합침.
    for key, value in override.items():
        # 양쪽 모두 그 키의 값이 dict라면 → 안쪽으로 한 단계 더 들어가 또 병합.
        # (자기 자신을 다시 부르는 "재귀". 중첩이 몇 단계든 끝까지 파고듦)
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            # 그 외에는 override 값으로 덮어씀(복사본을 넣어 원본과 분리).
            result[key] = deepcopy(value)
    return result   # 합쳐진 새 dict 반환


# ─────────────────────────────────────────────────────────────
# 여러 YAML 파일을 "순서대로" 읽어 하나의 설정으로 합치는 함수.
#   *paths = 인자를 몇 개든 받을 수 있다는 뜻. 예) load_configs("a.yaml", "b.yaml", "c.yaml")
#   뒤에 오는 파일이 앞 파일의 같은 설정을 덮어씁니다(우선순위가 더 높음).
# ─────────────────────────────────────────────────────────────
def load_configs(*paths: str | Path) -> dict[str, Any]:
    config: dict[str, Any] = {}        # 빈 dict에서 시작
    for path in paths:                 # 넘어온 파일 경로들을 앞에서부터 하나씩
        config = deep_merge(config, load_yaml(path))  # 읽어서 지금까지 결과 위에 병합
    return config                      # 전부 합쳐진 최종 설정 반환


# ─────────────────────────────────────────────────────────────
# 파이썬 dict 설정을 다시 YAML 파일로 저장하는 함수 (load의 반대).
# 보통 "이번 실험에 어떤 설정을 썼는지" 기록용 스냅샷을 남길 때 사용합니다.
#   -> None = 돌려주는 값 없이, 파일로 저장하는 "행동"만 하는 함수라는 뜻.
# ─────────────────────────────────────────────────────────────
def dump_yaml(config: dict[str, Any], path: str | Path) -> None:
    target = Path(path)
    # 저장할 폴더가 없으면 만들어 줌. parents=True(상위 폴더까지), exist_ok=True(이미 있어도 에러 안 냄).
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as f:   # "w" = 쓰기 모드(write)
        # safe_dump: dict를 YAML 텍스트로 변환해 파일에 기록.
        #   sort_keys=False  → 키를 알파벳순으로 재정렬하지 말고 원래 순서 유지
        #   allow_unicode=True → 한글 등을 \uXXXX로 깨뜨리지 말고 그대로 저장
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)

