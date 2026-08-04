# SCBench v2.2 results

This directory contains compact, machine-readable results from the frozen
SCBench v2.2 capability-11 panel. Full working directories, snapshots, prompts,
evaluator internals, generated code, and agent rollouts remain local and are
not published here.

The pinned panel uses `kkondaurov/scb-problems` `v1.0.2` at commit
`88e9666f9529c97a30951fca17cb38656ea0a5f1`, with catalog tree SHA256
`42a5a447856e057d67716edb83aec17ff72b98d50d83b500eb439fdc947c89e5`.
Pre-v2.2 observations are not comparable and are excluded.

| Run | Model | Reasoning | Sample | Strict | Current | Core | Tests | API-equivalent cost | Agent time | Wall time |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [2026-08-03 capability-11 sample 1](gpt-5.6-luna-xhigh-capability-11/20260803T141547/) | GPT-5.6 Luna | xhigh | 1 | 7/66 | 22/66 | 47/66 | 7,483/8,113 | $28.29 | 12:29:03.623 | 3:55:59.359 |
| [2026-08-03 capability-11 sample 1](gpt-5.6-terra-high-capability-11/20260803T182555/) | GPT-5.6 Terra | high | 1 | 10/66 | 24/66 | 49/66 | 7,516/8,113 | $31.59 | 5:10:00.298 | 2:01:33.945 |
| [2026-08-03 capability-11 sample 1](gpt-5.6-terra-xhigh-capability-11/20260803T205147/) | GPT-5.6 Terra | xhigh | 1 | 9/66 | 27/66 | 49/66 | 7,562/8,113 | $50.33 | 8:00:55.432 | 2:45:11.583 |
| [2026-08-03 capability-11 sample 1](gpt-5.6-sol-medium-capability-11/20260803T235211/) | GPT-5.6 Sol | medium | 1 | 12/66 | 24/66 | 50/66 | 7,508/8,113 | $60.73 | 5:10:40.867 | 2:20:39.558 |
| [2026-08-04 capability-11 sample 1](gpt-5.6-sol-high-capability-11/20260804T022121/) | GPT-5.6 Sol | high | 1 | 14/66 | 28/66 | 51/66 | 7,562/8,113 | $91.18 | 9:13:19.207 | 3:11:50.209 |
| [2026-08-04 capability-11 sample 1](gpt-5.6-sol-xhigh-capability-11/20260804T055236/) | GPT-5.6 Sol | xhigh | 1 | 18/66 | 31/66 | 52/66 | 7,645/8,113 | $116.46 | 11:40:34.407 | 4:06:21.631 |

## Preliminary sample-1 profile matrix

Every profile has one observation. The capability columns are outcomes, while
the reliability column below is limited to operational evaluator coverage; it
does not estimate model variance.

| Profile | Strict | Current | Core | Tests | API-equivalent cost | Agent time | Evaluator time | Wall time | Evaluator reliability |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Luna xhigh | 7/66 | 22/66 | 47/66 | 7,483/8,113 | $28.29 | 12:29:03.623 | 1:08:41.249 | 3:55:59.359 | 66/66, 0 failures |
| Terra high | 10/66 | 24/66 | 49/66 | 7,516/8,113 | $31.59 | 5:10:00.298 | 1:37:40.340 | 2:01:33.945 | 66/66, 0 failures |
| Terra xhigh | 9/66 | 27/66 | 49/66 | 7,562/8,113 | $50.33 | 8:00:55.432 | 1:09:02.881 | 2:45:11.583 | 66/66, 0 failures |
| Sol medium | 12/66 | 24/66 | 50/66 | 7,508/8,113 | $60.73 | 5:10:40.867 | 2:16:39.678 | 2:20:39.558 | 66/66, 0 failures |
| Sol high | 14/66 | 28/66 | 51/66 | 7,562/8,113 | $91.18 | 9:13:19.207 | 1:35:18.241 | 3:11:50.209 | 66/66, 0 failures |
| Sol xhigh | 18/66 | 31/66 | 52/66 | 7,645/8,113 | $116.46 | 11:40:34.407 | 1:08:03.176 | 4:06:21.631 | 66/66, 0 failures |

Sol xhigh is the preliminary capability leader, but it is also the most
expensive profile and used over 11 hours 40 minutes of recorded agent time.
Terra high has the best wall throughput and a much lower cost while retaining
most of the core score. Sol medium used nearly the same agent time as Terra
high for slightly more strict and core solves, but at almost twice the cost.
Luna xhigh was cheapest yet slow in agent time and weakest on capability.
Terra xhigh improved current behavior and tests over Terra high without a
strict or core gain. These are useful-speed trade-offs, not rankings with
measured stability; samples 2 and 3 remain deliberately paused.
