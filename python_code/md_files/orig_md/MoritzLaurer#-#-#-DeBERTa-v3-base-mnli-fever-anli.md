---
language:
- en
license: mit
tags:
- text-classification
- zero-shot-classification
datasets:
- multi_nli
- anli
- fever
metrics:
- accuracy
pipeline_tag: zero-shot-classification
model-index:
- name: MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
  results:
  - task:
      type: natural-language-inference
      name: Natural Language Inference
    dataset:
      name: anli
      type: anli
      config: plain_text
      split: test_r3
    metrics:
    - type: accuracy
      value: 0.495
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYWViYjQ5YTZlYjU4NjQyN2NhOTVhNjFjNGQyMmFiNmQyZjRkOTdhNzJmNjc3NGU4MmY0MjYyMzY5MjZhYzE0YiIsInZlcnNpb24iOjF9.S8pIQ7gEGokd_wKXMi6Bc3B2DThIP3cvVkTFErZ-2JxXTSCy1TBuulY3dzGfaiP7kTHbL52OuBhG_-wb7Ue9DQ
    - type: precision
      value: 0.4984740618243923
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiOTllZDU3NmVmYjk4ZmYzNjAwNzExMGZjNDMzOWRkZjRjMTRhNzhlZmI0ZmNlM2E0Mzk4OWE5NTM5MTYyYWU5NCIsInZlcnNpb24iOjF9.WHz_TUJgPVn-rU-9vBCDdmSMOuWzADwr09rJY6ktqRM46zytbyWs7Vcm7jqDrTkfU-rp0_7IyoNv_xEsKhJbBA
    - type: precision
      value: 0.495
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZjllODE3ZjUxZDhiMTI0MzZmYjY5OTUwYWI2OTc4ZjJhNTVjMjY2ODdkMmJlZjQ5YWQ1Mjk2ZThmYjJlM2RlYSIsInZlcnNpb24iOjF9.a9V06-O7l9S0Bv4vj0aard8128SAP61DZdXl_3XqdmNgt_C6KAoDBVueF2M2kF_kT6lRfEz6YW0ACIfJNXDYAA
    - type: precision
      value: 0.4984357572868885
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNjhiMzYzY2JiMmYwN2YxYzEwZTQ3NGI1NzFmMzliNjJkMDE2YzI5Njg1ZjEzMGIxODdiMDNmYmI4Y2Y2MmJkMiIsInZlcnNpb24iOjF9.xvZZaUMogw9MJjb3ls6h5liDlTqHMmNgqk6KbyDqQWfCcD255brCU3Xo6nECwaChS4te0dQu_iWGBqR_o2kYAA
    - type: recall
      value: 0.49461028192371476
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZDVjYTEzOTI0ZjVhOTk3ZTkzZmZhNTk5ODcxMWJhYWU4ZTRjYWVhNzcwOWY5YmI2NGFlYWE4NjM5MDY5NTExOSIsInZlcnNpb24iOjF9.xgHCB2rbCQBzHzUokw4u8JyOdhtF4yvPv1t8t7YiEkaAuM5MAPsVuCZ1VtlLapHS_IWetlocizsVl6akjh3cAQ
    - type: recall
      value: 0.495
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYTEyYmM0ZDQ0M2RiMDNhNjIxNzQ4OWZiNTBiOTAwZDFkNjNmYjBhNjA4NmQ0NjFkNmNiZTljNDkxNDg3NzIyYSIsInZlcnNpb24iOjF9.3FJPwNtwgFNvMjVxVAayaVXXR1sWlr0sqAYmXzmMzMxl7IJh6RS77dGPwFaqD3jamLVBiqPn9wsfz5lFK5yTAA
    - type: recall
      value: 0.495
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmY1MjZlZTQ4OTg5YzdlYmFhZDMzMmNlNjNkYmIyZGI4M2NjZjQ1ZDVkNmZkMTUxNjI3M2UwZmI1MDM1NDYwOSIsInZlcnNpb24iOjF9.cnbM6xjTLRa9z0wEDGd_Q4lTXVLRKIQ6_YLGLjf-t7Nto4lzxAeWF-RrwA0Mq9OPITlJq2Jk1Eg_0Utb13d9Dg
    - type: f1
      value: 0.4942810999491704
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2U3NGM1MDM4YTM4NzQxMGM4ZTIyZDM2YTQ1MGNlZWM1MzEzM2MxN2ZmZmRmYTM0OWJmZGJjYjM5OWEzMmZjNSIsInZlcnNpb24iOjF9.vMtge1F-tmMn9D3aVUuwcNEXjqpNgEyHAl9f5UDSoTYcOgTwi2vi5yRGRCl8y6Fx7BtgaCwMyoZVNbP5-GRtCA
    - type: f1
      value: 0.495
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNjBjMTQ5MmQ5OGE5OWJjZGMyNzg4N2RmNDUzMzQ5Zjc4ZTc4N2JlMTk0MTc2M2RjZTgzOTNlYWQzODAwNDI0NCIsInZlcnNpb24iOjF9.yxXG0CNWW8__xJC14BjbTY9QkXD75x6uCIXR51oKDemkP0b_xGyd-A2wPIuwNJN1EYkQevPY0bhVpRWBKyO9Bg
    - type: f1
      value: 0.4944671868893595
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzczNjQzY2FmMmY4NTAwYjNkYjJlN2I2NjI2Yjc0ZmQ3NjZiN2U5YWEwYjk4OTUyOTMzZTYyZjYzOTMzZGU2YiIsInZlcnNpb24iOjF9.mLOnst2ScPX7ZQwaUF12W2nv7-w9lX9-BxHl3-0T0gkSWnmtBSwYcL5faTX0_I5q33Fjz5tfkjpCJuxP5JYIBQ
    - type: loss
      value: 1.8788293600082397
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzRlOTYwYjU1Y2Y4ZGM0NDBjYTE2MmEzNWIwN2NiMWVkOWZlNzA2ZmQ3YjZjNzI4MjQwYWZhODIwMzU3ODAyZiIsInZlcnNpb24iOjF9._Xs9bl48MSavvp5eyamrP2iNlFWv35QZCrmWjJXLkUdIBx0ElCjEdxBb3dxPGnUxdpDzGMmOoKCPI44ZPXrtDw
  - task:
      type: natural-language-inference
      name: Natural Language Inference
    dataset:
      name: anli
      type: anli
      config: plain_text
      split: test_r1
    metrics:
    - type: accuracy
      value: 0.712
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYWYxMGY0ZWU0YTEyY2I3NmQwZmQ3YmFmNzQxNGU5OGNjN2ViN2I0ZjdkYWUzM2RmYzkzMDg3ZjVmNGYwNGZkZCIsInZlcnNpb24iOjF9.snWBusAeo1rrQqWk--vTxb-CBcFqM298YCtwTQGBZiFegKGSTSKzj-SM6HMNsmoQWmMuv7UfYPqYlnzEthOSAg
    - type: precision
      value: 0.7134839439315348
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNjMxMjg1Y2QwNzMwM2ZkNGM3ZTJhOGJmY2FkNGI1ZTFhOGQ3ODViNTJmZTYwMWJkZDYyYWRjMzFmZDI1NTM5YSIsInZlcnNpb24iOjF9.ZJnY6zYOBn-YEtN7uKzQ-VKXPwlIO1zq19Yuo37vBJNSs1dGDd8f1jgfdZuA19e_wA3Nc5nQKe9VXRwPHPgwAQ
    - type: precision
      value: 0.712
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZWM4YWQyODBlYTIwMWQxZDA1NmY1M2M2ODgwNDJiY2RhMDVhYTlkMDUzZTJkMThkYzRmNDg2YTdjMjczNGUwOCIsInZlcnNpb24iOjF9.SogsKHdbdlEs05IBYwXvlnaC_esg-DXAPc2KPRyHaVC5ItVHbxa63NpybSpao4baOoMlLG9aRe7TjG4gtB2dAQ
    - type: precision
      value: 0.7134676028447461
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODdjMzFkM2IwNWZiM2I4ZWViMmQ4NWM5MDY5ZWQxZjc1MGRmNjhmNzJhYWFmOWEwMjg3ZjhiZWM3YjlhOTIxNSIsInZlcnNpb24iOjF9._0JNIbiqLuDZrp_vrCljBe28xexZJPmigLyhkcO8AtH2VcNxWshwCpZuRF4bqvpMvnApJeuGMf3vXjCj0MC1Bw
    - type: recall
      value: 0.7119814425203647
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYjU4MWEyMzkyYzg1ZTIxMTc0M2NhMTgzOGEyZmY5OTg3M2Q1ZmMwNmU3ZmU1ZjA1MDk0OGZkMzM5NDVlZjBlNSIsInZlcnNpb24iOjF9.sZ3GTcmGGthpTLL7_Zovq8aBmE3Dp_PZi5v8ZI9yG9N6B_GjWvBuPC8ENXK1NwmwiHLsSvtKTG5JmAum-su0Dg
    - type: recall
      value: 0.712
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZDg3NGViZTlmMWM2ZDNhMzIzZGZkYWZhODQxNzg2MjNiNjQ0Zjg0NjQ1OWZkY2I5ODdiY2Y3Y2JjNzRmYjJkMiIsInZlcnNpb24iOjF9.bCZUzJamsozKWehnNph6E5coww5zZTrJdbWevWrSyfT0PyXc_wkZ-NKdyBAoqprBz3_8L3i5hPM6Qsy56b4BDA
    - type: recall
      value: 0.712
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMDk1MDJiOGUzZThlZjJjMzY4NjMzODFiZjUzZmIwMjIxY2UwNzBiN2IxMWEwMGJjZTkxODA0YzUxZDE3ODRhOCIsInZlcnNpb24iOjF9.z0dqvB3aBVYt3xRIb_M4svWebfQc0QaDFVFzHnlA5QGEHkHOW3OecGhHE4EzBqTDI3DASWZTGMjrMDDt0uOMBw
    - type: f1
      value: 0.7119226991285647
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiM2U0YjMwNzhmOTEyNDZhODU3MTU0YTM4MmQ0NzEzNWI1YjY0ZWQ3MWRiMTdiNTUzNWRkZThjMWE4M2NkZmI0MiIsInZlcnNpb24iOjF9.hhj1BXkuWi9wXrCjT9NwqaPETtOoYNiyqYsJEw-ufA8A4hVThKA6ZBtma1Q_M65-DZFfPEBDBNASLZ7EPSbmDw
    - type: f1
      value: 0.712
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODk0Y2EyMzc5M2ZlNWFlNDg2Zjc1OTQxNGY3YjA5YjUxYTYzZjRlZmU4ODYxNjA3ZjkxNGUzYjBmNmMxMzY5YiIsInZlcnNpb24iOjF9.DvKk-3hNh2LhN2ug5e0FgUntL3Ozdfl06Kz7jvmB-deOJH6INi2a2ZySXoEePoo8t2nR6ENFYu9QjMA2ojnpCA
    - type: f1
      value: 0.7119242267218338
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2MxOWFlMmI2NGRiMjkwN2Q5MWZhNDFlYzQxNWNmNzQ3OWYxZThmNDU2OWU1MTE5OGY2MWRlYWUyNDM3OTkzZCIsInZlcnNpb24iOjF9.QrTD1gE8_wRok9u59W-Mx0cX89K-h2Ad6qa8J5rmP8lc_rkG0ft2n5_GqH1CBZBJwMFYv91Pn6TuE3eGxJuUDA
    - type: loss
      value: 1.0105403661727905
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMmUwMTg4NjM3ZTBiZTIyODcyNDNmNTE5ZDZhMzNkMDMyNjcwOGQ5NmY0NTlhMjgyNmIzZjRiNDFiNjA3M2RkZSIsInZlcnNpb24iOjF9.sjBDVJV-jnygwcppmByAXpoo-Wzz178bBzozJEuYEiJaHSbk_xEevfJS1PmLUuplYslKb1iyEctnjI-5bl-XDw
  - task:
      type: natural-language-inference
      name: Natural Language Inference
    dataset:
      name: multi_nli
      type: multi_nli
      config: default
      split: validation_mismatched
    metrics:
    - type: accuracy
      value: 0.902766476810415
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjExZWM3YzA3ZDNlNjEwMmViNWEwZTE3MjJjNjEyNDhjOTQxNGFmMzBjZTk0ODUwYTc2OGNiZjYyMTBmNWZjZSIsInZlcnNpb24iOjF9.zbFAGrv2flpmweqS7Poxib7qHFLdW8eUTzshdOm2B9H-KWpIZCWC-P4p8TLMdNJnUcZJZ03Okil4qjIMqqIRCA
    - type: precision
      value: 0.9023816542652491
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2U2MGViNmJjNWQxNzRjOTkxNDIxZjZjNmM5YzE4ZjU5NTE5NjFlNmEzZWRlOGYxN2E3NTAwMTEwYjNhNzE0YSIsInZlcnNpb24iOjF9.WJjDJf56FROvf7Y5ShWnnxMvK_ZpQ2PibAOtSFhSiYJ7bt4TGOzMwaZ5RSTf_mcfXgRfWbXmy1jCwNhDb-5EAw
    - type: precision
      value: 0.902766476810415
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYzRhZTExOTc5NDczZjI1YmMzOGYyOTU2MDU1OGE5ZTczMDE0MmU0NzZhY2YzMDI1ZGQ3MGM5MmJiODFkNzUzZiIsInZlcnNpb24iOjF9.aRYcGEI1Y8-a0d8XOoXhBgsFyj9LWNwEjoIPc594y7kJn91wXIsXoR0-_0iy3uz41mWaTTlwJx7lI-kipFDvDQ
    - type: precision
      value: 0.9034597464719761
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMWQyMTZiZDA2OTUwZjRmNTFiMWRlZTNmOTliZmI2MWFmMjdjYzEyYTgwNzkyOTQzOTBmNTUyYjMwNTUxMTFkNiIsInZlcnNpb24iOjF9.hUtAMTl0THHUkaLcgk1Vy9IhjqJAXCJ_5STJ5A7k7s_SO9DHp3b6qusgwPmcGLYyPy1-j1dB2AIstxK4tHfmDA
    - type: recall
      value: 0.9024304801555488
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzAxZGJhNGI3ZDNlMjg2ZDIxNTgwMDY5MTFjM2ExZmIxMDBmZjUyNTliNWNkOGI0OTY3NTYyNWU3OWFlYTA3YiIsInZlcnNpb24iOjF9.1o_GNq8zmXa_50MUF_K63IDc2aUKNeUkNQ5fT592-SAo8WgiaP9Dh6bOEu2OqrpRQ57P4qm7OdJt7UKsrosMDA
    - type: recall
      value: 0.902766476810415
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZjhiMWE4Yjk0ODFkZjlkYjRlMjU1OTJmMjA2Njg1N2M4MzQ0OWE3N2FlYjY4NDgxZThjMmExYWQ5OGNmYmI1NSIsInZlcnNpb24iOjF9.Gmm5lf_qpxjXWWrycDze7LHR-6WGQc62WZTmcoc5uxWd0tivEUqCAFzFdbEU1jVKxQBIyDX77CPuBm7mUA4sCg
    - type: recall
      value: 0.902766476810415
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiY2EzZWYwNjNkYWE1YTcyZGZjNTNhMmNlNzgzYjk5MGJjOWJmZmE5NmYwM2U2NTA5ZDY3ZjFiMmRmZmQwY2QwYiIsInZlcnNpb24iOjF9.yA68rslg3e9kUR3rFTNJJTAad6Usr4uFmJvE_a7G2IvSKqLxG_pqsHszsWfg5mFBQLjWEAyCtdQYMdVayuYMBA
    - type: f1
      value: 0.9023086094638595
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMzMyMzZhNjI5MWRmZWJhMjkzN2E0MjM4ZTM5YzZmNTk5YTZmYzU4NDRiYjczZGQ4MDdhNjJiMGU0MjE3NDEwNyIsInZlcnNpb24iOjF9.RCMqH_xUMN97Vos54pTFfAMbLstXUMdFTs-eNaypbDb_Fc-MW8NLmJ6dzJsp9sSvhXyYjugjRMUpMpnQseKXDA
    - type: f1
      value: 0.902766476810415
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTYxZTZhZGM0NThlNTAzNmYwMTA4NDNkN2FiNzhhN2RlYThlYjcxMjE5MjBkMzhiOGYxZGRmMjE0NGM2ZWQ5ZSIsInZlcnNpb24iOjF9.wRfllNw2Gibmi1keU7d_GjkyO0F9HESCgJlJ9PHGZQRRT414nnB-DyRvulHjCNnaNjXqMi0LJimC3iBrNawwAw
    - type: f1
      value: 0.9030161011457231
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNDA0YjAxMWU5MjI4MWEzNTNjMzJlNjM3ZDMxOTE0ZTZhYmZlNmUyNDViNTU2NmMyMmM3MjAxZWVjNWJmZjI4MCIsInZlcnNpb24iOjF9.vJ8aUjfTbFMc1BgNUVpoVDuYwQJYQjwZQxblkUdvSoGtkW_AzQJ_KJ8Njc7IBA3ADgj8iZHjRQNIZkFCf-xICw
    - type: loss
      value: 0.3283354640007019
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiODdmYzYzNTUzZDNmOWIxM2E0ZmUyOWUzM2Y2NGRmZDNiYjg3ZTMzYTUyNzg3OWEzNzYyN2IyNmExOGRlMWUxYSIsInZlcnNpb24iOjF9.Qv0FzFZPkcBs9aHGf4TEREX4jdkc40NazdMlP2M_-w2wHwyjoAjvhk611RLXHcbicozNelZJLnsOMdEMnPLEDg
  - task:
      type: natural-language-inference
      name: Natural Language Inference
    dataset:
      name: anli
      type: anli
      config: plain_text
      split: dev_r1
    metrics:
    - type: accuracy
      value: 0.737
      name: Accuracy
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMTQ1ZGVkOTVmNTlhYjhkMjVlNTNhMjNmZWFjZWZjZjcxZmRhMDVlOWI0YTdkOTMwYjVjNWFlOGY4OTc1MmRhNiIsInZlcnNpb24iOjF9.wGLgKA1E46ljbLokdPeip_UCr1gqK8iSSbsJKX2vgKuuhDdUWWiECrUFN-bv_78JWKoKW5T0GF_hb-RVDzA0AQ
    - type: precision
      value: 0.737681071614645
      name: Precision Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmFkMGUwMjNhN2E3NzMxNTc5NDM0MjY1MGU5ODllM2Q2YzA1MDI3OGI1ZmI4YTcxN2E4ZDk5OWY2OGNiN2I0MCIsInZlcnNpb24iOjF9.6G5qhccjheaNfasgRyrkKBTaQPRzuPMZZ0hrLxTNzAydMDgx09FkFP3hni7WLRMWp0IpwzkEeBlxV-mPyQBtBw
    - type: precision
      value: 0.737
      name: Precision Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2QzYjQ4ZDZjOGU5YzI3YmFlMThlYTRkYTUyYWIyNzc4NDkwNzM1OWFiMTgyMzA0NDZmMGI3YTQxODBjM2EwMCIsInZlcnNpb24iOjF9.bvNWyzfct1CLJFx_EuD2GeKieVtyGJy0cwUBP2qJE1ey2i9SVn6n1Dr0AALTGBkxQ6n5-fJ61QFNufpdr2KvCA
    - type: precision
      value: 0.7376755842752241
      name: Precision Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiN2VmYWYzZWQwZmMzMDk0NTdlY2Y3NDkzYWY5ZTdmOGU0ZTUzZWE4YWFhZjVmODhkZmE1Njg4NjA5YjJmYWVhOSIsInZlcnNpb24iOjF9.50FQR2aoBpORLgYa7482ZTrRhT-KfIgv5ltBEHndUBMmqGF9Ru0LHENSGwyD_tO89sGPfiW32TxpbrNWiBdIBA
    - type: recall
      value: 0.7369675064285843
      name: Recall Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiZTM4OTAyNDYwNjY4Zjc5NDljNjBmNTg2Mzk4YjYxM2MyYTA0MDllYTMyNzEwOGI1ZTEwYWE3ZmU0NDZmZDg2NiIsInZlcnNpb24iOjF9.UvWBxuApNV3vd4hpgwqd6XPHCbkA_bB_Cw24ooquiOf0dstvjP3JvpGoDp5SniOzIOg3i2aYbcvFCLJqEXMZCQ
    - type: recall
      value: 0.737
      name: Recall Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiYmQ4MjMzNzRmNTI5NjIzNGQ0ZDFmZTA1MDU3OTk0MzYyMGI0NTMzZTZlMTQ1MDc1MzBkMGMzYjcxZjU1NDNjOSIsInZlcnNpb24iOjF9.kpbdXOpDG3CUB-kUEXsgFT3HWWIbu70wwzs2TNf0rhIuRrzdZz3dXXvwqu1BcLJTsOxl8G6NTiYXgnv-ul8lDg
    - type: recall
      value: 0.737
      name: Recall Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmU1ZWJkNWE0NjczY2NiZWYyNzYyMzllNzZmZTIxNWRkYTEyZDgxN2E0NTNmM2ExMTc1ZWVjMzBiYjg0ZmM1MiIsInZlcnNpb24iOjF9.S6HHWCWnut_LJqXbEA_Z8ZOTtyq6V51ZeiA0qbwzr0hapDYZOZHrN4prvSLvoNv-GiYDYKatwIsAZxCZc5fmCA
    - type: f1
      value: 0.7366853496239583
      name: F1 Macro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNzkxYmY2NTcyOTE0ZDdjNGY2ZmE4MzQwMGIxZTA2MDg1NzI5YTQ0MTdkZjdkNzNkMDM2NTk2MTNiNjU4ODMwZCIsInZlcnNpb24iOjF9.ECVaCBqGd0pnQT3xJF7yWrgecIb-5TMiVWpEO0MQGhYy43snkI6Qs-2FOXzvfwIWqG-Q6XIIhGbWZh5TFEGKCA
    - type: f1
      value: 0.737
      name: F1 Micro
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNDMwMWZiNzQyNWEzNmMzMDJjOTAxYzAxNzc0MTNlYzRkZjllYmNjZmU0OTgzZDFkNWM1ZWI5OTA2NzE5Y2YxOSIsInZlcnNpb24iOjF9.8yZFol_Gcj9n3w9Yk5wx48yql7p3wriDecv-6VSTAB6Q_MWLQAWsCEGRRhgGJ3zvhoRehJZdb35ozk36VOinDQ
    - type: f1
      value: 0.7366990292378379
      name: F1 Weighted
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiMjhhN2ZkMjc5ZGQ3ZGM1Nzk3ZTgwY2E1N2NjYjdhNjZlOTdhYmRlNGVjN2EwNTIzN2UyYTY2ODVlODhmY2Q4ZCIsInZlcnNpb24iOjF9.Cz7ClDAfCGpqdRTYd5v3dPjXFq8lZLXx8AX_rqmF-Jb8KocqVDsHWeZScW5I2oy951UrdMpiUOLieBuJLOmCCQ
    - type: loss
      value: 0.9349392056465149
      name: loss
      verified: true
      verifyToken: eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJoYXNoIjoiNmI4MTI5MDM1NjBmMzgzMzc2NjM5MzZhOGUyNTgyY2RlZTEyYTIzYzY2ZGJmODcxY2Q5OTVjOWU3OTQ2MzM1NSIsInZlcnNpb24iOjF9.bSOFnYC4Y2y2pW1AR-bgPUHKafR-0OHf8PvexK8eQLsS323Xy9-rYkKUaP09KY6_fk9GqAawv5eqj72B_uyeCA
---
# DeBERTa-v3-base-mnli-fever-anli
## Model description
This model was trained on the MultiNLI, Fever-NLI and Adversarial-NLI (ANLI) datasets, which comprise 763 913 NLI hypothesis-premise pairs. This base model outperforms almost all large models on the [ANLI benchmark](https://github.com/facebookresearch/anli). 
The base model is [DeBERTa-v3-base from Microsoft](https://huggingface.co/microsoft/deberta-v3-base). The v3 variant of DeBERTa substantially outperforms previous versions of the model by including a different pre-training objective, see annex 11 of the original [DeBERTa paper](https://arxiv.org/pdf/2006.03654.pdf). 

For highest performance (but less speed), I recommend using https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli.


### How to use the model
#### Simple zero-shot classification pipeline
```python
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
sequence_to_classify = "Angela Merkel is a politician in Germany and leader of the CDU"
candidate_labels = ["politics", "economy", "entertainment", "environment"]
output = classifier(sequence_to_classify, candidate_labels, multi_label=False)
print(output)
```
#### NLI use-case
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

premise = "I first thought that I liked the movie, but upon second thought it was actually disappointing."
hypothesis = "The movie was good."

input = tokenizer(premise, hypothesis, truncation=True, return_tensors="pt")
output = model(input["input_ids"].to(device))  # device = "cuda:0" or "cpu"
prediction = torch.softmax(output["logits"][0], -1).tolist()
label_names = ["entailment", "neutral", "contradiction"]
prediction = {name: round(float(pred) * 100, 1) for pred, name in zip(prediction, label_names)}
print(prediction)
```
### Training data
DeBERTa-v3-base-mnli-fever-anli was trained on the MultiNLI, Fever-NLI and Adversarial-NLI (ANLI) datasets, which comprise 763 913 NLI hypothesis-premise pairs.

### Training procedure
DeBERTa-v3-base-mnli-fever-anli was trained using the Hugging Face trainer with the following hyperparameters.
```
training_args = TrainingArguments(
    num_train_epochs=3,              # total number of training epochs
    learning_rate=2e-05,
    per_device_train_batch_size=32,   # batch size per device during training
    per_device_eval_batch_size=32,    # batch size for evaluation
    warmup_ratio=0.1,                # number of warmup steps for learning rate scheduler
    weight_decay=0.06,               # strength of weight decay
    fp16=True                        # mixed precision training
)
```
### Eval results
The model was evaluated using the test sets for MultiNLI and ANLI and the dev set for Fever-NLI. The metric used is accuracy.

mnli-m | mnli-mm | fever-nli | anli-all | anli-r3
---------|----------|---------|----------|----------
0.903 | 0.903 | 0.777 | 0.579 | 0.495

## Limitations and bias
Please consult the original DeBERTa paper and literature on different NLI datasets for potential biases. 

## Citation
If you use this model, please cite: Laurer, Moritz, Wouter van Atteveldt, Andreu Salleras Casas, and Kasper Welbers. 2022. ‘Less Annotating, More Classifying – Addressing the Data Scarcity Issue of Supervised Machine Learning with Deep Transfer Learning and BERT - NLI’. Preprint, June. Open Science Framework. https://osf.io/74b8k.

### Ideas for cooperation or questions?
If you have questions or ideas for cooperation, contact me at m{dot}laurer{at}vu{dot}nl or [LinkedIn](https://www.linkedin.com/in/moritz-laurer/)

### Debugging and issues
Note that DeBERTa-v3 was released on 06.12.21 and older versions of HF Transformers seem to have issues running the model (e.g. resulting in an issue with the tokenizer). Using Transformers>=4.13 might solve some issues. 

## Model Recycling

[Evaluation on 36 datasets](https://ibm.github.io/model-recycling/model_gain_chart?avg=0.65&mnli_lp=nan&20_newsgroup=-0.61&ag_news=-0.01&amazon_reviews_multi=0.46&anli=0.84&boolq=2.12&cb=16.07&cola=-0.76&copa=8.60&dbpedia=-0.40&esnli=-0.29&financial_phrasebank=-1.98&imdb=-0.47&isear=-0.22&mnli=-0.21&mrpc=0.50&multirc=1.91&poem_sentiment=1.73&qnli=0.07&qqp=-0.37&rotten_tomatoes=-0.74&rte=3.94&sst2=-0.45&sst_5bins=0.07&stsb=1.27&trec_coarse=-0.16&trec_fine=0.18&tweet_ev_emoji=-0.93&tweet_ev_emotion=-1.33&tweet_ev_hate=-1.67&tweet_ev_irony=-5.46&tweet_ev_offensive=-0.17&tweet_ev_sentiment=-0.11&wic=-0.21&wnli=-1.20&wsc=4.18&yahoo_answers=-0.70&model_name=MoritzLaurer%2FDeBERTa-v3-base-mnli-fever-anli&base_name=microsoft%2Fdeberta-v3-base) using MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli as a base model yields average score of 79.69 in comparison to 79.04 by microsoft/deberta-v3-base.

The model is ranked 2nd among all tested models for the microsoft/deberta-v3-base architecture as of 09/01/2023.

Results:

|   20_newsgroup |   ag_news |   amazon_reviews_multi |   anli |   boolq |      cb |    cola |   copa |   dbpedia |   esnli |   financial_phrasebank |   imdb |   isear |    mnli |    mrpc |   multirc |   poem_sentiment |   qnli |     qqp |   rotten_tomatoes |     rte |    sst2 |   sst_5bins |    stsb |   trec_coarse |   trec_fine |   tweet_ev_emoji |   tweet_ev_emotion |   tweet_ev_hate |   tweet_ev_irony |   tweet_ev_offensive |   tweet_ev_sentiment |     wic |    wnli |     wsc |   yahoo_answers |
|---------------:|----------:|-----------------------:|-------:|--------:|--------:|--------:|-------:|----------:|--------:|-----------------------:|-------:|--------:|--------:|--------:|----------:|-----------------:|-------:|--------:|------------------:|--------:|--------:|------------:|--------:|--------------:|------------:|-----------------:|-------------------:|----------------:|-----------------:|---------------------:|---------------------:|--------:|--------:|--------:|----------------:|
|        85.8072 |   90.4333 |                  67.32 | 59.625 |  85.107 | 91.0714 | 85.8102 |     67 |   79.0333 | 91.6327 |                   82.5 |  94.02 | 71.6428 | 89.5749 | 89.7059 |   64.1708 |          88.4615 | 93.575 | 91.4148 |           89.6811 | 86.2816 | 94.6101 |     57.0588 | 91.5508 |          97.6 |        91.2 |           45.264 |            82.6179 |         54.5455 |          74.3622 |              84.8837 |              71.6949 | 71.0031 | 69.0141 | 68.2692 |         71.3333 |


For more information, see: [Model Recycling](https://ibm.github.io/model-recycling/)
