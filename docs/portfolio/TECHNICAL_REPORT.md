# LangMani v1 技术报告

**副标题：可复现、可审计的语言条件机器人操作系统**

**Release：`v1.0.0`**

## 摘要

LangMani 在 ManiSkill 3.0.1 中构建了一个从自然语言到 Panda 机械臂操作的完整研究流水线：
确定性任务环境、特权运动规划专家、原始轨迹归档、LeRobotDataset v3 导出、八个 ACT
基线、共享策略语义诊断、模块化语言路由、安全拒绝和成对物理评测。系统支持三种颜色方块、
两个目标箱和六个 canonical TaskSpec，并以“同场景不同任务”的反事实设计隔离任务语义。

v1 的主要工程结论是：完整实验管线和物理验证可以通过，同时模型质量门可以诚实失败。
sealed final 中，Oracle 控制上限为 55/72（76.39%），NeuroSymbolic 端到端成功为 46/72
（63.89%）；61/72 指令被正确路由，11 条可执行指令被安全地假拒绝，15 个正确路由任务
在连续控制阶段超时。十类拒绝探针全部在 controller lookup、policy reset、environment reset
和 `env.step` 之前返回。最终 `final_pipeline_validated=true`、
`physical_target_validated=true`，但 `final_quality_gate_passed=false`，因此没有启动 SmolVLA。

## 1. 问题定义与设计目标

目标任务是 `LangMani-PickPlaceByInstruction-v0`：Panda 需要根据英文指令，从红、绿、蓝
三个方块中选择一个，并放入左或右目标箱。v1 不追求开放世界操作，而是用小而严格的任务空间
研究三个问题：语言是否能无泄漏地选择任务、控制是否能在反事实任务中保持语义一致、失败是否能
被准确归因。

核心约束包括：确定性 reset、批量 Torch 语义、真实视觉观察不包含 oracle target index、
原始数据可 action replay、训练只使用 train split、checkpoint 只由 validation 选择、test 在选择后
才开启、所有运行绑定 Git 和内容指纹。语言拒绝必须没有 TaskSpec，且必须在所有机器人运行时工作
之前返回。

## 2. 系统架构

系统分为五层：M1 提供物理任务与 typed metadata；M2 用特权状态生成专家演示；M3A/M3B
建立权威原始数据与派生训练数据；M4 建立冻结的 ACT 控制库；M5A 将自然语言转换为 TaskSpec
并只读选择对应控制器。Oracle EpisodeSpec 始终定义真实评测任务，预测 TaskSpec 只负责控制器选择，
因此 wrong-route 与 control failure 不会被混在一起。

拒绝路径与执行路径是并列的一等分支。`reject_*` 结果的 object/bin 字段必须为空；安全探针会验证
controller lookup、policy call、environment reset 和 `env.step` 都为零。执行路径保留 raw、binary、
projected、executed action 四种身份，避免把投影后的合法动作伪装成原始模型输出。

## 3. 数据与可复现性

M3A 使用 M2 专家采集 60 个完整 CounterfactualSceneGroup。每组固定一个物理 scene seed，包含
六个 TaskSpec；只有六条轨迹全部由专家成功、结构合法并通过真实 action replay 后，整组才被接纳。
最终 360 条 accepted episode 来自 65 个按序候选场景；5 组拒绝，404 次尝试全部入 manifest。

原始时间契约是 T actions、T terminated、T truncated、T labels 与 T+1 environment states。
M3B 从该权威归档恢复每个 pre-action state，重新渲染 256×256 `base_camera`，导出 64,548 帧，
并按 scene group 划分 288/36/36 episodes，保证同一物理布局不跨 train/validation/test。

所有持久身份使用 canonical serialization 与 SHA-256，而不是 Python `hash()` 或文件枚举顺序。
数据、checkpoint、处理器、任务映射、action space、运行配置和 Git commit 都进入 provenance。

## 4. 控制基线

M4 训练八个 100k-step ACT：六个 PerTask、一个 Mixed-Unconditioned 和一个 Oracle
Mixed-TaskOneHot。PerTask 在 locked test 达到 31/36，在 180 条 historical fresh-seed 上达到
143/180；无条件 mixed 只有 5/36 与 18/180；OneHot 达到 27/36 与 101/180。这证明同一观察对应
六种冲突动作时，无条件策略存在一对多问题，而显式 oracle task condition 能明显缓解冲突。

但完整的预设质量门没有全部通过，所以 `baseline_quality_validated=false`。后续 TaskToken 与
FactorFiLM 分别在 development 只得到 15/72 和 38/72，均被拒绝。共享 ACT 架构搜索因此关闭，
部署路径改为六个冻结 PerTask 策略加显式路由。

## 5. 语言路由与安全仲裁

M5A 先后比较规则路由器、紧凑分类器、Qwen3-1.7B、Qwen3-4B 和
NeuroSymbolicRouterV0。所有候选共享严格 RouterDecision schema、有限 reason vocabulary、最多
一次 repair，并保持 validation/development/final 隔离。失败候选冻结为 offline negative baseline，
不会因后续结果而修改 prompt、parser 或阈值。

sealed final 的 600 条语言记录显示：Qwen3-4B 的 status accuracy 最高（94.50%），但仍有
3.33% false route 且 schema-valid 仅 98%；NeuroSymbolic 的 false-route 为 0、schema-valid 为
100%，代价是 21.67% false rejection。安全仲裁降低了 unsafe route，却牺牲了 coverage，这成为
最终端到端差距的主要来源。

## 6. Sealed final 结果

sealed final 使用 12 个 unseen scene，每个 scene 六个任务，并对 Oracle 与 NeuroSymbolic 做
72 组严格初始状态配对。Oracle 成功 55/72，NeuroSymbolic 成功 46/72；46 组均成功、17 组均失败、
9 组仅 Oracle 成功。NeuroSymbolic 正确路由 61/72，剩余 11 条均为安全假拒绝；正确路由后的
15 个失败全部归因为 timeout。

没有 wrong-object route、wrong-bin route、off-table、arm projection、基础设施故障或 M2 调用。
夹爪维度投影率约为 Oracle 22.87%、NeuroSymbolic 23.08%，最大修正 0.05956；这些投影被单独
报告，没有隐藏在成功率中。

## 7. 四个代表性案例

**成功。** scene seed 2000000012，指令 “In this command, The desired result is the red cube
resting in the left bin.”。系统正确生成 red_cube→left_bin，选择匹配控制器；Oracle 与 learned
均成功，初始状态完全一致。

**安全假拒绝。** 同一 scene 的 blue_cube→left_bin 指令被拒绝。Oracle 成功，而 learned 没有
lookup/reset/step；这不是 unsafe behavior，却降低 coverage，是 sealed final 的核心质量缺口。

**正确路由后超时。** green_cube→left_bin 被正确路由到同一控制器，Oracle 与 learned 初始状态
一致且都超时，说明该失败属于冻结连续控制能力，而不是语言语义错误。

**安全拒绝。** “Push the red cube toward the left bin.” 被判为 `reject_unsupported` /
`unsupported_action`。object、bin、TaskSpec 全为空，controller lookup、policy reset、environment
reset 和 `env.step` 均为 0。

## 8. 验证与工程实践

项目将“实验正确执行”和“模型质量达到门槛”拆成独立布尔值。validation-only selection、locked
test、fresh-process reload、独立 verifier、原子 staging/promotion、checksum、resume、failure
taxonomy 和禁止源访问均进入结构化证据。硬件结果只有在 native Linux CUDA/Vulkan/PhysX
目标命令真实通过后才标记为 physical validation。

长时间运行采用持久 tmux、可恢复 checkpoint 和双 GPU 队列；源代码始终在本地完成、测试、提交、
推送，服务器只拉取干净提交。数据和输出从不进入 Git。M6 release 将结果、限制和案例整理为静态
材料，不再改变 v1 runtime。

## 9. 局限与负结果

v1 只覆盖六个 pick-and-place TaskSpec、固定几何和 canonical object/bin 语义。路由器的零 false
route 来源于保守仲裁，并没有达到所需 rejection taxonomy 与 coverage。PerTask 控制仍有 timeout，
夹爪输出经常需要显式投影。共享 ACT 的 TaskToken 与 FactorFiLM 都没有超过质量门；M5B
LatentGuard 只有结构实现，真实 probe 因缺少已接受 controller artifacts 而 blocked。

因此 v1 不声称开放词汇、跨任务泛化、端到端 foundation model 或真实世界部署能力。最重要的
研究产出是一个能够冻结失败、阻止越权阶段、并让语言、控制、动作边界和基础设施失败分别可见的
系统。

## 10. Release 结论

LangMani v1 的 release 判定是：工程管线完成且物理证据可信；最终模型质量未达预设门槛；不启动
SmolVLA；保留所有负结果。`v1.0.0` 固结 M0–M5A 与 M5B structural bridge 的源代码、契约、
结果索引和 portfolio 材料。未来研究必须从新版本或新里程碑开始，不能回写 v1 结果。
