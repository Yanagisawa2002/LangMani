# LangMani resume and interview kit

## 一句话项目介绍

LangMani 是一个基于 ManiSkill、LeRobot 与 ACT 的可复现语言条件机器人操作系统，覆盖确定性
环境、专家轨迹、反事实数据、策略训练、安全语言路由和独立物理验证，并将“实验有效”与“模型
质量达标”作为两个独立结论。

## 简历 bullet（中文）

- 从零搭建 LangMani 语言条件机器人操作流水线，打通 ManiSkill 3.0.1 环境、mplib 专家、
  ManiSkill-native 轨迹、LeRobotDataset v3、ACT 训练和自然语言安全路由；以 SHA-256、Git、
  原子 promotion 和独立 verifier 固结端到端 provenance。
- 设计 60 组反事实场景 × 6 TaskSpec 的 360-episode 数据集；全部轨迹通过真实 action replay，
  按 scene group 做 288/36/36 split，消除同布局跨 split 泄漏。
- 训练并评测 8 个 100k-step ACT 基线；PerTask 在 180 个 fresh-seed episodes 达到 143/180，
  对照 Mixed-Unconditioned 18/180 与 Oracle TaskOneHot 101/180，量化同场景多任务的一对多冲突。
- 构建零执行安全拒绝和 Oracle/learned 初始状态严格配对评测；sealed final 完成 600 条语言、
  144 个物理 rollouts 和 10 个拒绝探针，做到 0 wrong-object、0 wrong-bin route，并诚实冻结
  `final_quality_gate_passed=false` 的负结果。

## Resume bullets (English)

- Built an end-to-end, provenance-bound language-conditioned manipulation stack across ManiSkill
  3.0.1, mplib, LeRobotDataset v3, ACT, and strict language routing; separated pipeline validity,
  physical validity, and model-quality gates.
- Designed a 360-episode counterfactual dataset (60 physical scenes × 6 TaskSpecs), requiring expert
  success, structural validation, and real action replay for every accepted episode; enforced
  scene-group 288/36/36 train/validation/test isolation.
- Trained and evaluated eight 100k-step ACT baselines; achieved 143/180 fresh-seed successes with
  six PerTask policies versus 18/180 unconditioned and 101/180 oracle-conditioned mixed policies.
- Evaluated a safety-arbitrated language router on 600 commands and 72 paired physical tasks;
  obtained 46/72 end-to-end successes with zero wrong-object/wrong-bin routes and ten verified
  zero-runtime rejection probes, while preserving the failed final quality gate as a release result.

## 60 秒项目介绍

“这个项目研究的是语言条件机器人系统怎么做到可复现和可审计，而不只是跑出一个成功率。我先在
ManiSkill 里做了一个三方块、两目标箱的确定性环境，把 scene 和 task 分开，因此同一个物理布局
可以配六种指令。然后用 mplib 专家采集了 60 个完整反事实场景组，共 360 条轨迹；每条都必须
真实 action replay 才进入数据集。控制侧训练了八个 ACT 基线，PerTask 明显优于同场景混合无条件
策略。语言侧最终采用带安全仲裁的模块化路由，拒绝会在 controller lookup 和 env.step 前返回。
sealed final 的工程和物理验证通过，但 quality gate 没过：learned 46/72，对比 Oracle 55/72。
我保留了 11 次安全假拒绝和 15 次正确路由后 timeout，而不是重跑或放宽门槛。这套工作最有价值的
部分是证据边界、失败归因和可恢复实验工程。”

## 面试深挖问题

### 为什么不用一个端到端大模型？

六个 PerTask ACT 是 v1 中最可靠的连续控制。共享 ACT 的 TaskToken 与 FactorFiLM 都在预先锁定的
development gate 上失败。把语言路由与控制分开，可以单独验证 route、controller identity 和
control outcome，也能让 rejection 真正做到零执行。

### 为什么 360 条数据必须按 60 个完整组接纳？

反事实比较要求同一 scene seed 的六个任务拥有完全相同的初始物理状态。若只补齐单个失败 task，
数据就不再是完整组，任务分布和场景难度会产生选择偏差。

### 为什么最终失败仍然可以 release？

`final_pipeline_validated` 表示 schedule、证据、配对、执行和 verifier 正确；
`final_quality_gate_passed` 表示模型指标达到预设标准。两者拆开后，质量失败是有效实验结论，而不是
基础设施失败。v1 release 冻结了这个结论，并阻止自动进入 SmolVLA。

### 最难的工程问题是什么？

一是跨 ManiSkill、mplib、LeRobot 和 ACT 的版本/API 边界；二是长时间 GPU 训练与评估的安全恢复；
三是防止 validation/test 泄漏；四是保持 raw/projected/executed action、oracle/predicted TaskSpec、
route/control failure 等身份不被聚合结果掩盖。

### 下一步会做什么？

v1 不再修改。新的研究版本应优先解决两个独立瓶颈：降低安全仲裁的 false rejection，同时保持
zero unsafe route；以及提升冻结 PerTask 控制器在 hard scene 的 timeout 表现。任何 SmolVLA 或
LatentGuard 实验都需要新的明确授权、版本和 sealed schedule。
