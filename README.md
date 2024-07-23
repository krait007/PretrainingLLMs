# PretrainingLLMs

Deeplearning.AI & Upstage의 [Pretraining LLMs](https://www.deeplearning.ai/short-courses/pretraining-llms/?fbclid=IwZXh0bgNhZW0CMTAAAR39J5DLHa_XZO9YzmZ6jYfdXLfCDFBHT7vPaYZdkqg8nZR1uSM2hBXXBE8_aem_X3Dw6vq1xEppFeWv54Bxug) 강의를 들으며 작업한 노트북을 저장합니다.

**** M1 Mac 에서는 두 가지 문제 발생 ****

1. `torch.bfloat16` 학습 미지원 -> 미사용
2. `CustomDataset`, `CustomArguments`를 직접 선언하면 에러 발생 -> 별개 스크립트(util.py)로 분리 후 import하여 실행
