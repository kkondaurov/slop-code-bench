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
