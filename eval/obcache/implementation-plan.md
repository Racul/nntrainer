# OBCache 수정 계획

상태: 계획만 작성. 이 계획에 따른 제품 코드 수정·빌드·추가 측정은 시작하지 않았다.
기존 평가 결과와 바이너리는 보존한다. LongBench 제외 조건도 유지한다.

## 1. 목표 수식과 참조 검증

- 기준 논문은 [arXiv v4](https://arxiv.org/html/2510.07651v4)로 고정한다.
- Appendix A.5의 GQA 방식, 즉 query head별 점수 계산 후 같은 KV group에서
  합산하는 방식을 목표로 한다. 현재 포팅과 공식 코드 기본값 `pre_redc`는
  A/Z를 먼저 평균내므로, 이를 논문의 방식과 동일하다고 취급하지 않는다.
- 같은 Q/K/V로 점수·선택 index·압축 K/V를 독립 참조와 비교한다.
  현재 합성 kernel의 점수 일치만으로 논문 수식 일치를 주장하지 않는다.

## 2. 점수 계산 비용 개선

여기서 기존의 “분리 실험”은 점수 계산 함수만 떼어 비용을 측정한 진단이다.
추론 밖으로 점수 계산을 옮기거나 생략하는 알고리즘 변경을 뜻하지 않는다.
실제 추론 순서는 prefill → 점수 계산 → KV 선택·압축 → decode로 유지한다.

- dtype, stride, 데이터 포인터와 점수 종류를 원소 반복문 밖에서 결정한다.
- 같은 V의 half→float 변환을 query마다 반복하지 않고 재사용한다.
  변환 버퍼는 현재 레이어/필요한 block 범위로 제한하고 사용 후 해제한다.
- O 계산과 V/O 내적을 CPU 행렬 연산·SIMD로 처리하는 경로를 검토한다.
  관측 query를 작은 block으로 처리해 전체 N×N attention 버퍼는 추가하지 않는다.
- 먼저 참조 점수·선택 결과를 검증하고, 그다음 실제 prefill 비용을 측정한다.

기존 분리 실험은 합성 입력의 관측 query 8행을 사용했다. 3회 평균은
원본 1.818초, Tensor 접근/key 모드 선택 hoisting 1.127초,
V를 한 번 변환해 재사용한 조건 0.278초였다. 점수 차이는 0이었다.
이는 현재 `pre_redc` 계산의 진단이며 논문 GQA 검증이나 전체 모델 개선율이 아니다.

## 3. 실제 KV 저장소 축소 — 구현 범위에 포함

- prefill용 작업 K/V와 decode에 유지할 압축 K/V의 소유권을 분리한다.
  prefill 작업 저장소는 레이어별 수명으로 관리하는 방안을 검토한다.
- 선택한 K/V를 작은 실제 할당으로 복사하고 이후 생성 토큰의 추가 공간을 확보한다.
  cache budget은 보존 prompt token 수이며 decode token은 이후 추가된다.
- KVCacheManager의 소유 텐서와 그래프 cache placeholder/MHA 입력 view의
  shape·MemoryData를 함께 교체한다. 이전 할당을 가리키는 참조를 모두 해제한다.
  그래프 메모리 풀이 별도의 full-capacity KV를 계속 보유하는지도 확인한다.
- 절대 token/RoPE 위치와 물리 cache 용량 검사를 분리한다. 현재 `checkStep()`의
  논리 위치와 전체 capacity 비교를 작은 저장소에 그대로 적용하지 않는다.
- 미설정/zero-budget 경로와 기존 모델의 메모리 계약을 유지한다.

현재 조건의 KV payload는 4225행에서 약 462.1 MiB다. 보존 prompt 256행과
decode 128행을 담는 384행 payload는 약 42.0 MiB라는 산술값이다.
prefill 작업 공간·가중치·allocator 동작이 별도로 있으므로, 이를 전체 peak RSS의
예상 절감량으로 사용하지 않는다. 실제 예약 행 수는 생성 루프 경계 검증으로 확정한다.

## 4. 완료 확인 계획

- 다중 decode, 용량 경계, reset, K/V 대응, 절대 RoPE 위치를 검증한다.
- 할당 bytes와 이전 저장소 해제·재바인딩을 확인하는 검증을 추가한다.
- Linux/native Windows에서 prefill·점수 계산·decode 시간과,
  KV payload·전체 프로세스 peak·decode 중 resident memory를 구분한다.
- 새 바이너리와 새 결과 디렉터리로 baseline/방법론 비교를 수행한다.
  기존 72회 결과와 진단용 부분 실행을 새 측정에 섞지 않는다.

후속 실행은 사용자가 요청할 때 진행한다. 이 문서는 실행 완료 기록이 아니다.
