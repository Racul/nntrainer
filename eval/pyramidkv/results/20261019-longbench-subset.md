# LongBench v1 고정 부분 평가: 21개 태스크 × 첫 샘플 1개

**전체 데이터 평가 결과가 아니다.** 각 태스크의 고정 index 0 한 개를 양 조건·양 OS에서 greedy 1회 평가했다.
2026-10-19 00:01:47 KST에 완료했다. 총 84개 예측이 모두 유효하며, 조건별 태스크 표본 수는 1이다. 표준편차·신뢰구간은 추정하지 않는다.

|Task|Linux baseline|Linux PyramidKV|차이(pp)|Windows baseline|Windows PyramidKV|차이(pp)|
|---|---:|---:|---:|---:|---:|---:|
|narrativeqa|11.11|0.00|-11.11|11.11|0.00|-11.11|
|qasper|0.00|0.00|+0.00|0.00|0.00|+0.00|
|multifieldqa_en|66.67|66.67|+0.00|66.67|66.67|+0.00|
|multifieldqa_zh|50.00|50.00|+0.00|50.00|50.00|+0.00|
|hotpotqa|66.67|66.67|+0.00|66.67|66.67|+0.00|
|2wikimqa|0.00|0.00|+0.00|0.00|0.00|+0.00|
|musique|0.00|0.00|+0.00|0.00|0.00|+0.00|
|dureader|22.58|22.22|-0.36|22.58|22.22|-0.36|
|gov_report|38.98|13.38|-25.60|38.98|13.38|-25.60|
|qmsum|13.56|12.39|-1.17|13.56|12.39|-1.17|
|multi_news|16.99|13.95|-3.04|16.99|8.61|-8.38|
|vcsum|22.22|15.31|-6.91|22.22|15.31|-6.91|
|trec|100.00|100.00|+0.00|100.00|100.00|+0.00|
|triviaqa|100.00|100.00|+0.00|100.00|100.00|+0.00|
|samsum|44.44|44.44|+0.00|44.44|44.44|+0.00|
|lsht|0.00|0.00|+0.00|0.00|0.00|+0.00|
|passage_count|0.00|0.00|+0.00|0.00|0.00|+0.00|
|passage_retrieval_en|0.00|0.00|+0.00|0.00|0.00|+0.00|
|passage_retrieval_zh|0.00|0.00|+0.00|0.00|0.00|+0.00|
|lcc|7.00|7.00|+0.00|7.00|7.00|+0.00|
|repobench-p|53.00|68.00|+15.00|53.00|68.00|+15.00|

점수는 공식 태스크별 metric을 0~100으로 표시했다. 아래 평균은 선택된 21개 태스크 점수의 동일 가중치 평균이다.

- linux: baseline 29.20, PyramidKV 27.62, 차이 -1.58 pp; 기존 유효 예측 42개 재사용.
- windows: baseline 29.20, PyramidKV 27.37, 차이 -1.83 pp; 기존 유효 예측 29개 재사용.

입력 상한 8,192 tokens, 모델·정밀도·PyramidKV 설정·공식 prompt/scorer·generation 한도는 원래 평가와 같다.
샘플별 EOS로 실제 생성 길이가 다르므로, 품질 실행의 latency 비율을 동일 작업량의 speedup으로 해석하지 않는다.
입력별 latency/peak memory 비교는 별도 완료된 [72회 성능 실험](20261018-cpu.md)을 참조한다.
중단한 전체 실행의 예측 184개와 로그는 longbench-linux/에 그대로 보존했다.
태스크당 1개이므로 이 결과는 동작/경향 확인용이며 전체 LongBench 품질이나 논문 재현 성능을 대표하지 않는다.

독립 main baseline은 `7d14d001c9e70014de515dbd00ee6986d50d365f`,
PyramidKV 구현은 `2013bc91214286c0727bcdf0665b7954af1fef8e`이며,
품질 helper의 최종 버전은 `bbc3f99e32b959e6a439615158b74fd64530087d`다.
양 OS의 모델·실행 파일 hash는 앞선 성능 실험과 일치했다.
Qwen3-0.6B, FP32 weights/activations, FP16-bit UINT16 KV, CPU batch 1,
NNTR/OMP 4 threads, OpenBLAS 1 thread로 평가했다.
PyramidKV는 budget 256/window 64/kernel 5/beta 20이고, KV allocation은 유지한다.

공식 LongBench source revision은 `2e00731f8d0bff23dc4325161044d0ed8af94c1e`,
dataset revision은 `5e628be450b7e67fb7ae6e201bd6d8f7056f7672`다.
모든 태스크에서 원래 dataset index 0을 선택했다. 선택 identity SHA-256은
`86af973a2fb618e53ed193d814bab208a8c41d5b49d36ea181b5b448a7aaee43`다.
Official task prompt/metric/max_new_tokens와 middle truncation을 적용했다.
Qwen3의 non-thinking template을 사용하되 공식 plain-prompt 6개 태스크에는
chat wrapping을 하지 않았다. Greedy, seed 42, top_k 1, top_p 1, temperature 1이다.
첫 prefill 출력도 생성 한도에 포함하며, SAMSum은 공식 스크립트처럼 newline EOS를 추가한다.
Linux는 WSL2 Ubuntu, Windows는 native Windows 실행이다.

Windows SAMSum baseline의 추론은 정상 종료했지만, 원래 CRLF가 포함된 prompt가
CRT text 출력에서 CRCRLF로 변하면서 구형 파서의 정확한 입력 매칭이 실패했다.
원본 stdout을 bytes로 읽고 Windows의 LF-to-CRLF 변환을 되돌린 뒤 추출했다.
원래 실패 record와 로그를 보존하고, 성공 종료 marker·metrics·activation·정확한
prompt/trailer·공식 점수를 재검증해 새 record에 복구 provenance를 남겼다.
이 1건은 모델을 다시 실행하지 않았다. 다른 실패를 성공 처리하거나 제거하지 않았다.

최종 84건 모두 raw stdout/stderr와 예측을 재대조하고 공식 scorer로 다시 채점했다.
모든 PyramidKV 예측에서 28개 층의 activation/retention marker를 확인했고,
baseline에서는 다른 KV 방법론의 활성화가 없음을 확인했다.
Linux와 Windows의 `multi_news` PyramidKV 출력·점수가 다르며, 표에는 각각의 실측 점수를 유지했다.
모든 task model/controller 프로세스는 종료 확인했다. 전체 빌드 매트릭스 CI 통과를 주장하지 않는다.

[검증된 결과 JSON](20261019-longbench-subset.json)은 84건의 개별 점수·실제 token 수·시간,
stdout/stderr/예측 hash, 원본 manifest/records hash, 실행 protocol과 재사용·복구 provenance를 포함한다.
전체 prompt/정답 텍스트는 원본 local cases에 유지한다.
원본 로그와 중단 기록은 git에서 제외된 `build/validation/pyramidkv-run/`에 보존했다.
완료 manifest는 `longbench-subset-{linux,windows}-v2/manifest.json`,
원본 record는 같은 디렉터리의 `records.jsonl`이고, 현재 pipeline 상태는 `subset-pipeline-status.json`이다.
`final-longbench-audit.json`과 `final-process-check.json`에 마지막 검증을 기록했다.
