#!/bin/bash
# d15_gen_rollouts 全量提交脚本(审查修订 B1:去 array 化)。
# ┌─────────────────────────────────────────────────────────────────┐
# │ 门:冒烟 E 确认(E>=1.5,哨兵在案)且用户明确"放"之后才可执行。 │
# │ 本脚本本身即人工门的实体——不入任何自动化调用链。               │
# └─────────────────────────────────────────────────────────────────┘
# 提交 8 个独立作业(qsub -v SHARD_ID=$s),替代 PBS array;
# 弃 array 依据见 pbs/d15_gen_rollouts.pbs 头注释(6 月模板无可靠先例
# 存证 + 独立作业便于单 shard 重提/止损)。
# 单 shard 重提:qsub -v SHARD_ID=<s> -N d15gen_s<s> pbs/d15_gen_rollouts.pbs
set -euo pipefail
cd "$(dirname "$0")/.."
for s in $(seq 0 7); do
    qsub -v SHARD_ID=${s} -N d15gen_s${s} pbs/d15_gen_rollouts.pbs
done
